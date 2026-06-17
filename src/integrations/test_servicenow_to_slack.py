"""
test_servicenow_to_slack.py
=============================
Manual smoke test: fetch a ServiceNow incident, post it to Slack.

All credentials are read from environment variables — never hardcoded.

Run from project root:
    python -m integrations.test_servicenow_to_slack
"""

import os
import dotenv

from servicenow_client import ServiceNowClient
from slack_client import SlackClient

dotenv.load_dotenv()

# ============================================================
# Read all configuration from environment — fail loudly and
# clearly if anything required is missing, rather than
# proceeding with None values that produce confusing errors
# further down the line.
# ============================================================

try:
    INSTANCE_URL    = os.environ["SERVICENOW_INSTANCE_URL"]
    USERNAME        = os.environ["SERVICENOW_USERNAME"]
    PASSWORD        = os.environ["SERVICENOW_PASSWORD"]
    INCIDENT_NUMBER = os.environ.get("SERVICENOW_TEST_INCIDENT", "INC0010002")
    WEBHOOK_URL     = os.environ["SLACK_WEBHOOK_URL"]
except KeyError as e:
    print(f"Missing required environment variable: {e}")
    print("Set SERVICENOW_INSTANCE_URL, SERVICENOW_USERNAME, "
          "SERVICENOW_PASSWORD, SLACK_WEBHOOK_URL in your .env file.")
    raise SystemExit(1)


# ============================================================
# Create Clients
# ============================================================

snow_client = ServiceNowClient(
    instance_url=INSTANCE_URL,
    username=USERNAME,
    password=PASSWORD
)

slack_client = SlackClient(
    webhook_url=WEBHOOK_URL
)


# ============================================================
# Fetch Incident
# ============================================================

incident = snow_client.get_incident_by_number(
    INCIDENT_NUMBER
)


# ============================================================
# Build Slack Message
# ============================================================

if incident:

    message = f"""
🚨 New Incident Detected

Incident Number:
{incident.get('number')}

Short Description:
{incident.get('short_description')}

Description:
{incident.get('description')}

Priority:
{incident.get('priority')}

Category:
{incident.get('category')}
"""

    slack_client.send_message(message)

else:
    print("Incident not found.")
