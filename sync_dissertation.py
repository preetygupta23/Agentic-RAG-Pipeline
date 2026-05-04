import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('calendar', 'v3', credentials=creds)

def add_task(service, summary, date_str):
    # Morning Slot: 11 AM - 1 PM | Afternoon Slot: 2 PM - 6 PM
    slots = [("11:00:00", "13:00:00", "AM"), ("14:00:00", "18:00:00", "PM")]
    for start, end, label in slots:
        event = {
            'summary': f"[{label}] {summary}",
            'start': {'dateTime': f'{date_str}T{start}', 'timeZone': 'Asia/Kolkata'},
            'end': {'dateTime': f'{date_str}T{end}', 'timeZone': 'Asia/Kolkata'},
            'reminders': {'useDefault': False, 'overrides': [{'method': 'popup', 'minutes': 15}]},
        }
        service.events().insert(calendarId='primary', body=event).execute()

if __name__ == '__main__':
    service = get_service()
    # Adding the Phase 1 Kickoff as an example
    add_task(service, "Dissertation Kickoff: Repo Setup", "2026-04-25")
    print("✅ Schedule synced to Google Calendar.")