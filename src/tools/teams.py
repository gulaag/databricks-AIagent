"""
Microsoft Teams incoming webhook integration.

Sends a formatted Adaptive Card payload to a Teams channel via
a pre-configured incoming webhook URL. No OAuth or Graph API required.
"""

import json
import requests
import mlflow


@mlflow.trace(name="post_to_teams", span_type="TOOL")
def post_to_teams(agenda_content: str, webhook_url: str) -> str:
    """Post a formatted agenda to a Microsoft Teams channel via incoming webhook.

    Args:
        agenda_content: The plain-text or markdown agenda body to be posted.
        webhook_url: The Teams incoming webhook URL (stored in UC secrets at runtime).

    Returns:
        A status string indicating success or the error message on failure.

    Raises:
        Does not raise; all exceptions are caught and returned as error strings
        so the agent orchestrator can decide how to handle failures gracefully.
    """
    adaptive_card_payload = {
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
                            "text": agenda_content,
                            "wrap": True,
                            "spacing": "Medium",
                        },
                    ],
                },
            }
        ],
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url=webhook_url,
            headers=headers,
            data=json.dumps(adaptive_card_payload),
            timeout=10,
        )
        response.raise_for_status()
        return f"SUCCESS: Message posted to Teams. HTTP {response.status_code}."

    except requests.exceptions.Timeout:
        return "ERROR: Teams webhook request timed out after 10 seconds."

    except requests.exceptions.HTTPError as exc:
        return (
            f"ERROR: Teams webhook returned HTTP {exc.response.status_code}: "
            f"{exc.response.text[:200]}"
        )

    except requests.exceptions.RequestException as exc:
        return f"ERROR: Network error posting to Teams: {str(exc)}"

    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Unexpected failure in post_to_teams: {str(exc)}"
