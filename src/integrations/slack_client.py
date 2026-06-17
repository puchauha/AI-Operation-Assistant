# ============================================================
# File: slack_client.py
#
# Purpose:
# Send messages to Slack channel using Incoming Webhook
# ============================================================

import requests


class SlackClient:

    def __init__(self, webhook_url: str):

        self.webhook_url = webhook_url

    def send_message(self, message: str):

        payload = {
            "text": message
        }

        response = requests.post(
            self.webhook_url,
            json=payload
        )

        response.raise_for_status()

        print("Message sent to Slack successfully.")