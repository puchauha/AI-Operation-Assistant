# ============================================================
# File: slack_client.py
#
# Purpose:
# Slack API client for the AI Operations Assistant.
#
# This file has been updated from the original single-method
# version to support per-incident channels. It now has two
# distinct ways to send messages:
#
#   1. send_message() — still uses the original Incoming
#      Webhook URL for backward compatibility with any existing
#      integrations. Kept in place because the webhook is still
#      a valid, simpler mechanism for general notifications.
#
#   2. post_to_channel() — uses the Bot Token (xoxb-...) to
#      post a message to any specific channel by its channel_id.
#      This is what the per-incident channel flow uses — the
#      bot creates a channel, then posts the Decision Card to
#      that exact channel's ID.
#
# New methods for per-incident channel management:
#   3. create_incident_channel() — creates a new public Slack
#      channel named after the incident number
#   4. invite_bot_to_channel() — adds the bot itself as a
#      member of the new channel so it can post and receive
#      events there
#
# Why two token types:
#   The Incoming Webhook (SLACK_WEBHOOK_URL in .env) only works
#   for one fixed, pre-configured channel. It cannot create
#   channels or post to arbitrary channel IDs — it is too
#   limited for the per-incident channel pattern.
#
#   The Bot Token (SLACK_BOT_TOKEN in .env, starts with xoxb-)
#   is a full API token that can do anything the bot has been
#   granted permission to do — create channels, post anywhere,
#   invite users. This is what the Slack Web API uses.
#
# Slack Web API base URL:
#   https://slack.com/api/<method_name>
#   All methods return JSON with {"ok": true/false, ...}
#   We check "ok" rather than just the HTTP status code because
#   Slack returns HTTP 200 for API-level errors too, putting
#   the real success/failure info inside the JSON body.
# ============================================================

import requests


class SlackClient:

    def __init__(self, webhook_url: str = None, bot_token: str = None):
        """
        Initialize the Slack client.

        Parameters
        ----------
        webhook_url : str, optional
            Incoming Webhook URL (starts with https://hooks.slack.com/).
            Required only if you use send_message().

        bot_token : str, optional
            Bot User OAuth Token (starts with xoxb-).
            Required for create_incident_channel(),
            invite_bot_to_channel(), and post_to_channel().
        """
        self.webhook_url = webhook_url
        self.bot_token   = bot_token

        # Base URL for the Slack Web API — all Bot Token API calls
        # use this base with different method names appended.
        self._api_base = "https://slack.com/api"

        # Standard headers for all Bot Token API calls.
        # "Authorization: Bearer xoxb-..." is how the Slack Web API
        # authenticates requests. Without this header, every API
        # call returns {"ok": false, "error": "not_authed"}.
        if bot_token:
            self._api_headers = {
                "Authorization": f"Bearer {bot_token}",
                "Content-Type":  "application/json; charset=utf-8",
            }

    # ============================================================
    # Original webhook-based method — unchanged
    # ============================================================

    def send_message(self, message: str):
        """
        Send a plain text message via the Incoming Webhook.
        Only works for the single fixed channel the webhook
        was configured for. Kept for backward compatibility.
        """
        if not self.webhook_url:
            raise ValueError(
                "webhook_url is required for send_message(). "
                "Set SLACK_WEBHOOK_URL in your .env file."
            )
        payload  = {"text": message}
        response = requests.post(self.webhook_url, json=payload)
        response.raise_for_status()
        print("Message sent via webhook successfully.")

    # ============================================================
    # New Bot Token methods for per-incident channel management
    # ============================================================

    def create_incident_channel(self, incident_number: str) -> str:
        """
        Create a new public Slack channel named after the incident.

        Slack channel names must be lowercase, with no spaces —
        we convert the incident number (e.g. INC0010002) to
        lowercase (inc0010002) to meet this requirement.

        Parameters
        ----------
        incident_number : str
            The ServiceNow incident number, e.g. "INC0010002".
            The channel will be named "inc-0010002" (using a
            hyphen to separate the prefix from the number, which
            is more readable than "inc0010002" as a channel name).

        Returns
        -------
        str
            The new channel's channel_id (e.g. "C0B69EZTZL4").
            This ID is what you use in all subsequent API calls
            to post to or reference this specific channel —
            NOT the human-readable channel name.

        Raises
        ------
        RuntimeError
            If the channel already exists or if the API call fails.
        """
        if not self.bot_token:
            raise ValueError(
                "bot_token is required for create_incident_channel(). "
                "Set SLACK_BOT_TOKEN in your .env file."
            )

        # Build the channel name from the incident number.
        # "INC0010002" → "inc-0010002"
        # Lower-case is required by Slack. The hyphen makes it
        # more readable in the channel list sidebar.
        channel_name = incident_number.lower().replace("inc", "inc-", 1)

        response = requests.post(
            f"{self._api_base}/conversations.create",
            headers=self._api_headers,
            json={"name": channel_name, "is_private": False},
        )

        data = response.json()

        # Slack returns HTTP 200 even for errors — we must check
        # the "ok" field inside the JSON body, not just the HTTP
        # status code, to know if the call actually succeeded.
        if not data.get("ok"):
            error = data.get("error", "unknown_error")

            # "name_taken" means this channel already exists —
            # this happens if the same incident somehow gets
            # processed twice. In that case, we look up the
            # existing channel's ID rather than failing entirely.
            if error == "name_taken":
                print(f"  [INFO] Channel {channel_name} already exists — "
                      f"looking up its ID")
                return self._get_channel_id_by_name(channel_name)

            raise RuntimeError(
                f"Failed to create Slack channel '{channel_name}': {error}"
            )

        channel_id = data["channel"]["id"]
        print(f"  ✅ Created Slack channel #{channel_name} (ID: {channel_id})")
        return channel_id

    def _get_channel_id_by_name(self, channel_name: str) -> str:
        """
        Look up a channel's ID by its name. Used as a fallback
        when create_incident_channel() finds the channel already
        exists. Private helper — not part of the public API.
        """
        response = requests.get(
            f"{self._api_base}/conversations.list",
            headers=self._api_headers,
            params={"limit": 200},
        )
        data = response.json()

        if not data.get("ok"):
            raise RuntimeError(
                f"Failed to list channels: {data.get('error')}"
            )

        for channel in data.get("channels", []):
            if channel["name"] == channel_name:
                return channel["id"]

        raise RuntimeError(
            f"Channel '{channel_name}' not found even after name_taken error"
        )

    def invite_bot_to_channel(self, channel_id: str) -> None:
        """
        Invite the bot itself to join a channel.

        Even though the bot created the channel, Slack does not
        automatically make it a member — the bot needs to be
        explicitly invited before it can receive message events
        from that channel. Without this step, messages typed in
        the new channel would NOT trigger our slack_listener.py
        event handler.

        Parameters
        ----------
        channel_id : str
            The channel_id returned by create_incident_channel().
        """
        if not self.bot_token:
            raise ValueError(
                "bot_token is required for invite_bot_to_channel()."
            )

        # First, get the bot's own user ID — we need this to
        # invite "itself" to the channel it just created.
        # auth.test is a simple API call that returns info about
        # the token being used, including the bot's user_id.
        auth_response = requests.post(
            f"{self._api_base}/auth.test",
            headers=self._api_headers,
        )
        auth_data  = auth_response.json()
        bot_user_id = auth_data.get("user_id")

        if not bot_user_id:
            raise RuntimeError(
                f"Could not determine bot user ID: {auth_data.get('error')}"
            )

        # Now invite that user ID to the channel.
        response = requests.post(
            f"{self._api_base}/conversations.invite",
            headers=self._api_headers,
            json={"channel": channel_id, "users": bot_user_id},
        )
        data = response.json()

        if not data.get("ok"):
            error = data.get("error", "unknown_error")
            # "already_in_channel" is not a real error — the bot
            # is already a member, which is exactly what we want.
            if error != "already_in_channel":
                raise RuntimeError(
                    f"Failed to invite bot to channel {channel_id}: {error}"
                )

        print(f"  ✅ Bot invited to channel {channel_id}")

    def post_to_channel(self, channel_id: str, message: str) -> None:
        """
        Post a message to a specific channel using the Bot Token.

        Unlike send_message() (which uses a fixed webhook URL),
        this method can post to ANY channel by its ID — which is
        exactly what per-incident channels require.

        Parameters
        ----------
        channel_id : str
            The Slack channel ID to post to (e.g. "C0B69EZTZL4").
            Use the channel_id returned by create_incident_channel(),
            NOT the human-readable channel name.

        message : str
            The message text to post. Supports Slack's mrkdwn
            formatting: *bold*, `code`, ```code block```, etc.
        """
        if not self.bot_token:
            raise ValueError(
                "bot_token is required for post_to_channel()."
            )

        response = requests.post(
            f"{self._api_base}/chat.postMessage",
            headers=self._api_headers,
            json={"channel": channel_id, "text": message},
        )
        data = response.json()

        if not data.get("ok"):
            raise RuntimeError(
                f"Failed to post message to channel {channel_id}: "
                f"{data.get('error')}"
            )

        print(f"  ✅ Message posted to channel {channel_id}")

    def invite_user_to_channel(self, channel_id: str, user_id: str) -> None:
        """
        Invite a specific human user to an incident channel.

        This is how the human agent gets automatically added to the
        dedicated channel created for each incident — without this,
        the agent would have to manually browse and join every new
        channel, which breaks the real-time collaboration flow.

        MVP approach: the user_id is read from SLACK_DEFAULT_AGENT_USER_ID
        in .env — one hardcoded user who receives all incidents.

        Production upgrade path (Phase 2):
            UserResolver.get_slack_id(servicenow_username) returns the
            correct user_id for whichever agent was assigned the ticket
            by the least-loaded algorithm. The call site in
            incident_poller.py does not change — only the value of
            user_id passed in changes.

        Parameters
        ----------
        channel_id : str
            The Slack channel_id to invite the user to.

        user_id : str
            The Slack user ID of the human agent to invite.
            Starts with "U", e.g. "U0B65SC463C".
            Find yours in Slack: click your profile picture →
            Profile → three-dot menu → Copy member ID.
        """
        if not self.bot_token:
            raise ValueError(
                "bot_token is required for invite_user_to_channel()."
            )

        response = requests.post(
            f"{self._api_base}/conversations.invite",
            headers=self._api_headers,
            json={"channel": channel_id, "users": user_id},
        )
        data = response.json()

        if not data.get("ok"):
            error = data.get("error", "unknown_error")
            # "already_in_channel" is not a real error — the user
            # is already a member, which is fine.
            if error != "already_in_channel":
                # Log but don't crash — a failed invite is not
                # critical enough to stop the investigation pipeline.
                # The human can always join manually via Browse Channels.
                print(f"  [WARN] Could not invite user {user_id} to "
                      f"channel {channel_id}: {error}")
                return

        print(f"  ✅ User {user_id} invited to channel {channel_id}")

