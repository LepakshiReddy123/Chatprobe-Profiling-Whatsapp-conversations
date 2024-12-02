import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

st.sidebar.title("Chat Probe: Profiling WhatsApp Conversations")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    try:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
         # byte data into string
        data=bytes_data.decode("utf-8")
         # st.text(data) to get display of text data
        df=preprocessor.preprocess(data)
         #to display data into dataframe
        if df.empty:
            st.error("The uploaded file is empty or could not be processed.")
        else:
            st.dataframe(df)
    except Exception as e:
        st.error(f"Error processing file: {e}")
    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)
    if st.sidebar.button("Show Analysis"):
        # Stats Area
        num_messages,words,num_media_messages,num_links,num_deleted_messages = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4, col5= st.columns(5)
        with col1:
             st.header("Total Messages")
             st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)
        with col5:
            st.header("Deleted Messages")
            st.title(num_deleted_messages)
        # daily timeline
        # Filter out unwanted messages
        filtered_df = helper.filter_messages(df)
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user,filtered_df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='blue')
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Message Count', fontweight='bold')
        plt.title('Daily Timeline')
        plt.xticks(rotation='vertical', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        # Weekly Timeline
        st.title("Weekly Timeline")
        weekly_timeline_data = helper.weekly_timeline(selected_user, filtered_df)
        # Plot weekly timeline
        fig, ax = plt.subplots()
        ax.plot(weekly_timeline_data['week_start'], weekly_timeline_data['message_count'], color='purple')
        plt.xlabel('Week', fontweight='bold')
        plt.ylabel('Message Count', fontweight='bold')
        plt.title('Weekly Timeline', fontweight='bold')
        plt.xticks(rotation='vertical', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, filtered_df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xlabel('Month', fontweight='bold')
        plt.ylabel('Message Count', fontweight='bold')
        plt.title('Monthly Timeline', fontweight='bold')
        plt.xticks(rotation='vertical',)
        plt.tight_layout()
        st.pyplot(fig)


        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(filtered_df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                c = ['r', 'b', 'g', 'm', 'y']
                ax.bar(x.index, x.values, color=c,width=0.5)
                plt.xlabel('User Name', fontweight='bold')
                plt.ylabel('Message Count', fontweight='bold')
                plt.title('Most Busy User', fontweight='bold')
                plt.xticks(rotation='vertical')
                plt.tight_layout()
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df, width=400)




        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, filtered_df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple',width=0.5)
            plt.xlabel('Day Name', fontweight='bold')
            plt.ylabel('Message Count', fontweight='bold')
            plt.title('Most busy day', fontweight='bold')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, filtered_df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange',width=0.5)
            plt.xlabel('Month Name', fontweight='bold')
            plt.ylabel('Message Count', fontweight='bold')
            plt.title('Most Busy Month', fontweight='bold')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            st.pyplot(fig)
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, filtered_df)
        fig, ax = plt.subplots()
        plt.tight_layout()
        plt.title('Weekly Activity', fontweight='bold')

        # Customize the color bar
        sns.heatmap(user_heatmap, cbar_kws={'label': 'Activity Count'}, cmap='Paired')
        st.pyplot(fig)

        # Perform sentiment analysis
        df = helper.perform_sentiment_analysis(filtered_df)

        # Plot sentiment trends over time
        st.title("Sentiment Trends Over Time")
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['sentiment_score'], marker='d',linestyle='-',color='blue', markerfacecolor='magenta')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Sentiment Score', fontweight='bold')
        plt.title("Sentiment Analysis", fontweight='bold')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, filtered_df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc, interpolation='bilinear')
        ax.axis('off')
        plt.title('Word Cloud',fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        #most common words
        most_common_df = helper.most_common_words(selected_user, filtered_df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1],color='g')
        plt.xticks(rotation='vertical')
        st.title('Most Common Words')
        plt.tight_layout()
        st.pyplot(fig)

        #Topic Modelling
        st.title("Topic Modeling and Visualization")
        # Perform topic modeling
        num_topics = st.sidebar.slider("Number of Topics", min_value=3, max_value=10, value=10)
        @st.cache_data
        def get_topic_modeling_result(num_topics):
            lda_output, topic_keywords = helper.topic_modeling(filtered_df, num_topics=num_topics)
            return lda_output, topic_keywords
        lda_output, topic_keywords = get_topic_modeling_result(num_topics)

        # Display topic keywords
        st.subheader("Topic Keywords")
        for i, keywords in enumerate(topic_keywords):
            st.write(f"Topic {i + 1}: {', '.join(keywords)}")

        # Visualize topic distributions
        topic_distribution_df = pd.DataFrame(lda_output, columns=[f"Topic {i + 1}" for i in range(num_topics)])
        st.subheader("Topic Distributions")
        st.write(topic_distribution_df)

        # Plot topic distributions
        st.subheader("Topic Distribution Plot")
        fig, ax = plt.subplots()
        sns.barplot(data=topic_distribution_df,ax=ax, width=0.5)
        plt.xticks(rotation='vertical')
        plt.xlabel('Topic Distribution',fontweight='bold')
        plt.ylabel('percentage',fontweight='bold')
        plt.title('Topic Modelling',fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        # Create interaction graph
        interaction_graph = helper.create_interaction_graph(filtered_df)
        # Analyze network centrality
        centrality_measures = helper.analyze_network_centrality(interaction_graph)
        # Display centrality measures
        st.subheader("Network Centrality Measures")
        for measure, values in centrality_measures.items():
            st.write(f"{measure}:")
            # Convert centrality values to a DataFrame for easier plotting
            centrality_df = pd.DataFrame(values.items(), columns=['Participant', measure])
            centrality_df.set_index('Participant', inplace=True)

            # Plot centrality values
            st.bar_chart(centrality_df)

        # Analyze emotions
        emotions = helper.analyze_emotions(df['message'])
        # Create a DataFrame from emotion analysis results
        emotion_df = pd.DataFrame(emotions)
        # Visualize emotions using a bar chart
        st.subheader("Emotions Expressed in Messages")
        st.bar_chart(emotion_df)

        # Emoji analysis
        selected_user = st.sidebar.selectbox('Select User', ['Overall'] + df['user'].unique().tolist())

        st.title("Emoji Analysis")

        emoji_df = helper.emoji_helper(selected_user, filtered_df)

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:
            # Create a pie chart
            fig, ax = plt.subplots()
            patches, texts, _ = ax.pie(emoji_df['count'].head(), labels=emoji_df['emoji'].head(), autopct="%0.2f",
                                       startangle=90)

            # Set custom colors for each pie slice
            colors = plt.cm.viridis(np.linspace(0, 1, len(emoji_df['emoji'].head())))
            for patch, color in zip(patches, colors):
                patch.set_color(color)

            # Create a legend with emoji labels
            emoji_patches = [mpatches.Patch(color=color, label=f"{emoji}") for emoji, color in
                             zip(emoji_df['emoji'].head(), colors)]
            ax.legend(handles=emoji_patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')

            # Show the pie chart
            st.pyplot(fig)