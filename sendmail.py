def mail(to, text):
    import smtplib  
    server = smtplib.SMTP_SSL( "smtp.gmail.com", 465 )
    server.login( "your email", "password" )
    server.sendmail(to, text )
    server.quit()

