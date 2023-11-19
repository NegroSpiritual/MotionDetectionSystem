import pushbullet
import secrets
def send_push_notification(api_key, title, body):
    pb = pushbullet.Pushbullet(api_key)
    push = pb.push_note(title, body)
    print(f"Notification sent with push ID {push['iden']}")

if __name__ == "__main__":
    api_key = secrets.PUSHBULLET_API_KEY
    
    # Customize the notification title and body
    notification_title = 'Alert!'
    notification_body = 'This is a test notification from your Python script.'

    send_push_notification(api_key, notification_title, notification_body)
