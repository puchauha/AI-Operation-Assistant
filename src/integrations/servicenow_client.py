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
