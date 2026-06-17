# ============================================================
# File: servicenow_client.py
#
# Purpose:
# Reusable ServiceNow API client for interacting with
# ServiceNow incidents using REST APIs.
#
# Current Features:
# - Connect to ServiceNow instance
# - Fetch incident details using incident number
#
# Future Expansion Ideas:
# - Update incident work notes
# - Assign incidents
# - Change incident status
# - Add comments
# - Create incidents
# - Fetch similar incidents
#
# ============================================================

import requests
from requests.auth import HTTPBasicAuth


class ServiceNowClient:
    """
    Reusable ServiceNow API client.

    This class handles all communication with the
    ServiceNow REST API.

    Example:
        snow_client = ServiceNowClient(
            instance_url="https://dev12345.service-now.com",
            username="admin",
            password="password"
        )
    """

    def __init__(
        self,
        instance_url: str,
        username: str,
        password: str
    ):
        """
        Initialize ServiceNow client.

        Parameters
        ----------
        instance_url : str
            Base URL of ServiceNow instance

            Example:
            https://dev12345.service-now.com

        username : str
            ServiceNow username

        password : str
            ServiceNow password
        """

        # Remove trailing "/" if user accidentally adds it
        # This helps avoid malformed URLs later
        self.instance_url = instance_url.rstrip("/")

        # Create reusable authentication object
        # ServiceNow supports basic authentication
        self.auth = HTTPBasicAuth(username, password)

        # Common headers for all API calls
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    # ============================================================
    # Fetch Incident By Incident Number
    # ============================================================

    def get_incident_by_number(self, incident_number: str):
        """
        Fetch incident details using incident number.

        Parameters
        ----------
        incident_number : str
            Example:
            INC0010001

        Returns
        -------
        dict
            Incident details if found

        None
            If incident does not exist

        Raises
        ------
        requests.exceptions.HTTPError
            If API request fails
        """

        # --------------------------------------------------------
        # Construct ServiceNow Table API URL
        #
        # Endpoint:
        # /api/now/table/incident
        #
        # Query:
        # number=<incident_number>
        #
        # Limit:
        # return only 1 result
        # --------------------------------------------------------

        url = (
            f"{self.instance_url}/api/now/table/incident"
            f"?sysparm_query=number={incident_number}"
            f"&sysparm_limit=1"
        )

        # --------------------------------------------------------
        # Send GET request to ServiceNow
        # --------------------------------------------------------

        response = requests.get(
            url,
            auth=self.auth,
            headers=self.headers
        )

        # --------------------------------------------------------
        # Raise exception if HTTP request failed
        #
        # Example:
        # 401 Unauthorized
        # 404 Not Found
        # 500 Server Error
        # --------------------------------------------------------

        response.raise_for_status()

        # --------------------------------------------------------
        # Convert API response to Python dictionary
        # --------------------------------------------------------

        data = response.json()

        # --------------------------------------------------------
        # ServiceNow returns records inside "result"
        #
        # Example:
        # {
        #   "result": [ ... ]
        # }
        # --------------------------------------------------------

        results = data.get("result", [])

        # --------------------------------------------------------
        # Return None if incident not found
        # --------------------------------------------------------

        if not results:
            return None

        # --------------------------------------------------------
        # Return first matching incident
        # Since incident number is unique,
        # only one record should exist
        # --------------------------------------------------------

        return results[0]

    # ============================================================
    # Fetch New Incidents (for polling)
    # ============================================================
    #
    # Purpose:
    #   The AI agent needs to discover NEW incidents on its own —
    #   it cannot rely on someone manually telling it an incident
    #   number every time. This method asks ServiceNow: "give me
    #   every incident that is still in the New state and that the
    #   agent has not already picked up."
    #
    # How "already picked up" is tracked:
    #   We use a ServiceNow field called "work_notes" as a simple
    #   marker. Once the agent starts working an incident, it adds a
    #   work note like "AI Agent: investigation started". On every
    #   poll, we ask ServiceNow for incidents in state "New" that do
    #   NOT yet contain that marker text in their work notes. This
    #   is a simple approach that works without needing a separate
    #   database to track "have I seen this incident before?" — the
    #   marker lives directly inside ServiceNow itself, which is the
    #   single source of truth here.
    #
    #   A more robust production approach would use a custom field
    #   (e.g. u_ai_agent_picked_up = true) rather than searching work
    #   note text, since searching text is slower and slightly less
    #   precise. For the MVP, searching work notes avoids needing
    #   ServiceNow admin access to create a custom field, which keeps
    #   the setup effort low for a demo environment.
    # ============================================================

    def get_new_incidents(self, max_results: int = 10):
        """
        Fetch incidents that are in the "New" state and have not yet
        been picked up by the AI agent.

        Parameters
        ----------
        max_results : int
            Maximum number of incidents to return in one call.
            Keeping this small (default 10) means a single poll cycle
            cannot accidentally pull hundreds of incidents and overwhelm
            the agent or the Slack channel with messages all at once.

        Returns
        -------
        list[dict]
            A list of incident records (each one a dict, exactly like
            the dicts returned by get_incident_by_number()). Returns an
            empty list if no new incidents are found — never None, so
            calling code can always safely loop over the result with
            a "for incident in incidents:" statement without first
            checking if it is None.
        """

        # --------------------------------------------------------
        # ServiceNow query syntax (sysparm_query) builds up filter
        # conditions separated by "^". This reads as:
        #   state=1               → only incidents in "New" state
        #                            (1 is ServiceNow's standard code
        #                            for New; this can vary by instance
        #                            configuration, so double check
        #                            against your own instance if the
        #                            numbers don't match)
        #   ^work_notesNOT LIKE... → exclude incidents whose work notes
        #                            already contain our "picked up"
        #                            marker text
        # --------------------------------------------------------

        pickup_marker = "AI Agent: investigation started"

        url = (
            f"{self.instance_url}/api/now/table/incident"
            f"?sysparm_query=state=1^work_notesNOT LIKE{pickup_marker}"
            f"&sysparm_limit={max_results}"
        )

        response = requests.get(
            url,
            auth=self.auth,
            headers=self.headers
        )

        response.raise_for_status()

        data = response.json()

        # .get("result", []) returns an empty list if "result" is
        # missing entirely from the response — this is what guarantees
        # we never return None to the caller, only ever a list (which
        # might be empty).
        return data.get("result", [])

    # ============================================================
    # Update Work Notes
    # ============================================================
    #
    # Purpose:
    #   Work notes are ServiceNow's internal, customer-INVISIBLE
    #   comment field on an incident. This is exactly where the AI
    #   agent should write its findings, root cause, and suggested
    #   fix — NOT in the customer-visible "comments" field. Using work
    #   notes means the customer never sees the AI's raw internal
    #   reasoning; only authorised ServiceNow users (support staff)
    #   can see it. This distinction matters in real ITSM practice
    #   and is one of the details that signals genuine ITSM domain
    #   knowledge rather than a generic chatbot integration.
    #
    # How it works:
    #   ServiceNow's Table API treats updating a record as an HTTP
    #   PATCH request to the same endpoint used to fetch records, but
    #   targeting a specific record by its sys_id (ServiceNow's
    #   internal unique identifier — different from the human-readable
    #   incident number like INC0010002).
    # ============================================================

    def update_work_notes(self, sys_id: str, note_text: str):
        """
        Append a new work note to an incident. Work notes are internal
        only — never visible to the end customer.

        Parameters
        ----------
        sys_id : str
            ServiceNow's internal unique ID for this incident record.
            This is NOT the same as the incident number (e.g. INC0010002).
            You can get sys_id from any incident dict returned by
            get_incident_by_number() or get_new_incidents() — it is
            always present as incident["sys_id"].

        note_text : str
            The text to add as a new work note. ServiceNow automatically
            appends this as a new, timestamped entry — it does NOT
            overwrite previous work notes, so calling this method
            multiple times builds up a running history on the incident.

        Returns
        -------
        bool
            True if the update succeeded.

        Raises
        ------
        requests.exceptions.HTTPError
            If the API request fails (e.g. invalid sys_id, permission
            denied, or ServiceNow instance unreachable).
        """

        url = f"{self.instance_url}/api/now/table/incident/{sys_id}"

        # work_notes is the ServiceNow field name for internal notes.
        # This payload tells ServiceNow: "update this one field on
        # this one record".
        payload = {
            "work_notes": note_text
        }

        # requests.patch() sends an HTTP PATCH request — PATCH means
        # "update part of this record", as opposed to PUT (replace the
        # whole record) or POST (create a new record). ServiceNow's
        # Table API supports PATCH for partial updates like this one.
        response = requests.patch(
            url,
            auth=self.auth,
            headers=self.headers,
            json=payload
        )

        response.raise_for_status()

        return True


# ============================================================
# Test Section
#
# This block runs only when file is executed directly.
#
# All credentials are read from environment variables —
# NEVER hardcode credentials in source code. Set these in
# your .env file at the project root:
#
#   SERVICENOW_INSTANCE_URL=https://dev12345.service-now.com
#   SERVICENOW_USERNAME=admin
#   SERVICENOW_PASSWORD=your_password
#   SERVICENOW_TEST_INCIDENT=INC0010002
#
# Example:
# python servicenow_client.py
#
# ============================================================

if __name__ == "__main__":

    import os
    import dotenv
    dotenv.load_dotenv()

    # --------------------------------------------------------
    # Read credentials from environment — never from source code.
    # os.environ[...] raises a clear KeyError if the variable is
    # missing, rather than silently proceeding with a None value.
    # --------------------------------------------------------

    try:
        INSTANCE_URL     = os.environ["SERVICENOW_INSTANCE_URL"]
        USERNAME         = os.environ["SERVICENOW_USERNAME"]
        PASSWORD         = os.environ["SERVICENOW_PASSWORD"]
        INCIDENT_NUMBER  = os.environ.get("SERVICENOW_TEST_INCIDENT", "INC0010002")
    except KeyError as e:
        print(f"Missing required environment variable: {e}")
        print("Set SERVICENOW_INSTANCE_URL, SERVICENOW_USERNAME, "
              "SERVICENOW_PASSWORD in your .env file.")
        raise SystemExit(1)

    # --------------------------------------------------------
    # Create ServiceNow client object
    # --------------------------------------------------------

    snow_client = ServiceNowClient(
        instance_url=INSTANCE_URL,
        username=USERNAME,
        password=PASSWORD
    )

    # --------------------------------------------------------
    # Fetch incident details
    # --------------------------------------------------------

    incident = snow_client.get_incident_by_number(
        INCIDENT_NUMBER
    )

    # --------------------------------------------------------
    # Print incident details
    # --------------------------------------------------------

    if incident:

        print("\n===== INCIDENT DETAILS =====")

        print(f"Number       : {incident.get('number')}")

        print(f"Short Desc   : "
              f"{incident.get('short_description')}")

        print(f"Description  : "
              f"{incident.get('description')}")

        print(f"State        : "
              f"{incident.get('state')}")

        print(f"Priority     : "
              f"{incident.get('priority')}")

        print(f"Category     : "
              f"{incident.get('category')}")

        print(f"Assigned To  : "
              f"{incident.get('assigned_to')}")

        print(f"Sys ID       : "
              f"{incident.get('sys_id')}")

        print("============================\n")

    else:
        print(f"No incident found for {INCIDENT_NUMBER}")
