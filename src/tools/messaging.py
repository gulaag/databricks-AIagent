"""
Chat channel posting tool (Slack or Microsoft Teams).

Posts a formatted announcement to an incoming webhook. The payload format is
chosen from the webhook URL so the same tool works for either platform:
  - Slack incoming webhooks  -> {"text": ...}
  - Teams incoming webhooks  -> Adaptive Card message
No OAuth or Graph/Slack app is required — just the webhook URL, supplied at
runtime from a Databricks secret.
"""

import json

import mlflow
import requests


_SLACK_SECTION_LIMIT = 2900  # Slack mrkdwn section text limit is 3000; stay under it.


def _chunk(text: str, size: int = _SLACK_SECTION_LIMIT) -> list[str]:
    """Split text into <=size pieces so each fits one Slack section block."""
    return [text[i : i + size] for i in range(0, len(text), size)] or [" "]


def _slack_payload(message: str) -> dict:
    """Build a Slack incoming-webhook Block Kit payload (polished card layout)."""
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "📢 Tech Engineer 勉強会 — 案内",
                "emoji": True,
            },
        }
    ]
    for chunk in _chunk(message):
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": chunk}})
    blocks.append({"type": "divider"})
    blocks.append(
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "🤖 Databricks Action Agent が自動生成しました"}
            ],
        }
    )
    # `text` is the notification fallback shown in previews / older clients.
    return {"text": message[:3000], "blocks": blocks}


def _teams_payload(message: str) -> dict:
    """Build a Teams incoming-webhook Adaptive Card payload."""
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": "Tech Engineer Study Group — Session Announcement",
                            "weight": "Bolder",
                            "size": "Large",
                            "wrap": True,
                        },
                        {
                            "type": "TextBlock",
                            "text": message,
                            "wrap": True,
                            "spacing": "Medium",
                        },
                    ],
                },
            }
        ],
    }


@mlflow.trace(name="post_to_channel", span_type="TOOL")
def post_to_channel(message: str, webhook_url: str) -> str:
    """Post an announcement to a Slack or Teams channel via an incoming webhook.

    The platform is detected from the webhook URL (``hooks.slack.com`` -> Slack,
    otherwise Teams), and the matching payload format is sent.

    Args:
        message: The plain-text / markdown announcement body to post.
        webhook_url: The incoming webhook URL (supplied from a secret at runtime).

    Returns:
        A status string indicating success or the error message on failure.

    Raises:
        Does not raise; all exceptions are caught and returned as error strings
        so the agent orchestrator can decide how to handle failures gracefully.
    """
    if not webhook_url:
        return "ERROR: No webhook URL configured; cannot post."

    is_slack = "hooks.slack.com" in webhook_url
    platform = "Slack" if is_slack else "Teams"
    payload = _slack_payload(message) if is_slack else _teams_payload(message)
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url=webhook_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=10,
        )
        response.raise_for_status()
        return f"SUCCESS: Message posted to {platform}. HTTP {response.status_code}."

    except requests.exceptions.Timeout:
        return f"ERROR: {platform} webhook request timed out after 10 seconds."

    except requests.exceptions.HTTPError as exc:
        return (
            f"ERROR: {platform} webhook returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:200]}"
        )

    except requests.exceptions.RequestException as exc:
        return f"ERROR: Network error posting to {platform}: {str(exc)}"

    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Unexpected failure in post_to_channel: {str(exc)}"
