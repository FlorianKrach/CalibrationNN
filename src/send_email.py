"""
author: Florian Krach
"""

import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.Utils import formatdate


def send_email(mailadress='fkrach@student.ethz.ch', smtpserver='mail.ethz.ch', port=587, username='fkrach',
               password='', to = 'florian.krach33@gmail.com',
               subject = 'python mail', body = "DONE"):

    From = mailadress
    msg = MIMEMultipart()
    msg['From'] = From
    msg['To'] = to
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    text = msg.as_string()

    server = smtplib.SMTP(smtpserver, port)
    server.ehlo()  # Has something to do with sending information
    server.starttls()  # Use encrypted SSL mode
    server.ehlo()  # To make starttls work
    server.login(username, password)
    failed = server.sendmail(From, to, text)
    server.quit()

    return failed

# print send_email()

