import re
import pandas as pd

def preprocess(data):
    pattern_24hr = '\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s-\s'
    if re.match(pattern_24hr,data):
        pattern=pattern_24hr
        messages=re.split(pattern,data)[2:]
        dates=re.findall(pattern,data)[1:]
    else:
        pattern = r'\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s(?:AM|PM)\s-\s'
        messages = re.split(pattern, data)[2:]
        dates = re.findall(pattern, data)[1:]
        dates = [date.replace('\u202f', '') for date in dates]
        time_pattern = r'(\d{1,2}:\d{2})(?:AM|PM)'
        for i, date_time in enumerate(dates):
            time_match = re.search(time_pattern, date_time)
            if time_match:
                time_str = time_match.group(1)
                hours, minutes = map(int, time_str.split(':'))
                if 'PM' in date_time:
                    if hours != 12:
                        hours += 12
                else:
                    if hours == 12:
                        hours = 0
                time_24hr = f'{hours:02}:{minutes:02}'
                dates[i] = re.sub(time_pattern, time_24hr, date_time)
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = df['message_date'].str.replace(' - ', '')
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M')
    df.rename(columns={'message_date': 'date'}, inplace=True)
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            user = entry[1]
            if user != 'group_notification':
                users.append(user)
                messages.append(entry[2:])
        else:
            users.append('group_notification')
            messages.append(entry[0])
    df['user'] = users
    df['message'] = messages
    if 'message' in df.columns:
        df['message'] = df['message'].astype(str)
    df.drop(columns=['user_message'], inplace=True)
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()

    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['period'] = period
    # Add this code after loading and preprocessing the chat data
    df['receiver'] = df['user'].shift(-1)  # Assuming the next message sender is the receiver
    df = remove_group_notification(df)
    return df


def remove_group_notification(df):
    # Remove rows with user 'group_notification' and reset index
    df = df[df['user'] != 'group_notification'].reset_index(drop=True)
    return df