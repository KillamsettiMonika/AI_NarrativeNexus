def display_topics(model, feature_names, no_top_words=15):
    """
    Display top words per topic for LDA/NMF
    """
    for idx, topic in enumerate(model.components_):
        print(f"\nTopic {idx}: ",
              " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
def display_topics(model, feature_names, no_top_words=15):
    """
    Display top words per topic for LDA/NMF
    """
    for idx, topic in enumerate(model.components_):
        print(f"\nTopic {idx}: ",
              " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
