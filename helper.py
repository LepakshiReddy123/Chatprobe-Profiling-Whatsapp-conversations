from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
extract = URLExtract()
def filter_messages(df):
    # Define specific patterns to ignore
    ignore_patterns = [
        'You deleted this message',
        'This message was deleted',
        'null',
        'Media omitted'
    ]
    # Filter out messages containing specific patterns
    return df[~df['message'].apply(lambda x: isinstance(x, str) and any(pattern in x for pattern in ignore_patterns))]
def fetch_stats(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    # Filter out specific messages
    filtered_df = filter_messages(df)
    # Fetch the number of messages
    num_messages = len(filtered_df)
    # fetch the total number of words
    words = []
    for message in filtered_df['message']:
        if isinstance(message, str):
         words.extend(message.split(' '))
    # fetch number of media messages
    df['message'] = df['message'].str.lower().str.strip()
    num_media_messages = df[df['message'].str.contains('<Media omitted>', case=False)].shape[0]
    # fetch number of links shared
    links = []
    for message in df['message']:
       if isinstance(message, str):
          links.extend(extract.find_urls(message))
    # Count occurrences of deleted messages patterns
    df['message'] = df['message'].str.lower().str.strip()
    num_deleted_messages = df[df['message'].str.contains('You deleted this message|This message was deleted', case=False)].shape[0]
    return num_messages, len(words), num_media_messages, len(links), num_deleted_messages
def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline
def weekly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df=df[df['user'] == selected_user]
    # Group by week and count messages
    weekly_counts = df.groupby(df['date'].dt.strftime('%Y-%U')).size().reset_index(name='message_count')
     # Parse week-year string to datetime
    weekly_counts['week_start'] = pd.to_datetime(weekly_counts['date'] + '-0', format='%Y-%U-%w')
    return weekly_counts
def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline
def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
            columns={'user': 'name', 'count': 'percent'})
    return x, df
def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()
def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()
def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap
def perform_sentiment_analysis(df):
    sentiment_scores = []
    for message in df['message']:
        if isinstance(message, str):
            blob = TextBlob(message)
            sentiment_scores.append(blob.sentiment.polarity)
        else:
            sentiment_scores.append(None)  # Handle non-string messages
    df['sentiment_score'] = sentiment_scores
    return df

def create_wordcloud(selected_user,df):
    f = open('stop words.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)
    # Filter out non-string messages
    df=df[df['message'].apply(lambda x: isinstance(x, str))]
    temp['message']=temp['message'].apply(remove_stop_words)
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    df_wc = wc.generate(' '.join(df['message']))
    return df_wc
def most_common_words(selected_user,df):
    f= open('stop words.txt','r')
    stop_words=f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []
    for message in temp['message']:
        if isinstance(message, str):
            for word in message.lower().split():
                if word not in stop_words:
                    words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def topic_modeling(df, num_topics=5):
    # Filter out unwanted messages
    filtered_df = filter_messages(df)
    # Convert filtered messages to list
    messages = filtered_df['message'].tolist()
    # Initialize CountVectorizer
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    # Fit and transform the messages
    X = vectorizer.fit_transform(messages)
    # Initialize LDA model
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    # Fit LDA model to the data
    lda_output = lda_model.fit_transform(X)
    # Get the keywords associated with each topic
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = get_topic_keywords(lda_model, feature_names, num_words=10)
    return lda_output, topic_keywords
# Topic modeling with LDA

def get_topic_keywords(model, feature_names, num_words=10):
    topic_keywords = []
    for topic_weights in model.components_:
        top_keyword_indices = topic_weights.argsort()[:-num_words - 1:-1]
        topic_keywords.append([feature_names[i] for i in top_keyword_indices])
    return topic_keywords
def create_interaction_graph(df):
    G = nx.DiGraph()
    # Add edges between participants
    for index, row in df.iterrows():
        sender = row['user']
        receiver = row['receiver']  # Assuming 'receiver' is a column indicating the recipient of the message
        if pd.notnull(receiver) and sender != 'group_notification' and receiver != 'group_notification':
            G.add_edge(sender, receiver)
    return G
def analyze_network_centrality(G):
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    # Combine centrality measures into a dictionary
    centrality_measures = {
        'Degree Centrality': degree_centrality,
        'Betweenness Centrality': betweenness_centrality
    }
    return centrality_measures
def analyze_emotions(messages):
    analyzer = SentimentIntensityAnalyzer()
    emotion_results = []
    for message in messages:
        sentiment_score = analyzer.polarity_scores(message)
        emotion_results.append(sentiment_score)
    return emotion_results




# Define emoji unicode ranges
emoji_ranges = [
    (0x1F601, 0x1F64F),  # Emoticons
    (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
    (0x1F680, 0x1F6FF),  # Transport and Map Symbols
    (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    (0x2600, 0x26FF),    # Miscellaneous Symbols
    (0x2700, 0x27BF),    # Dingbats
    (0xFE00, 0xFE0F),    # Variation Selectors
    (0x1F1E6, 0x1F1FF)   # Flags (iOS)
]

# Function to check if a character is an emoji
def is_emoji(char):
    return any(start <= ord(char) <= end for start, end in emoji_ranges)

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if is_emoji(c)])

    emoji_counts = Counter(emojis)

    emoji_df = pd.DataFrame(emoji_counts.items(), columns=['emoji', 'count']).sort_values(by='count', ascending=False)
    # Reset index to have it in ascending order
    emoji_df.reset_index(drop=True, inplace=True)
    return emoji_df








