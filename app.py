import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Streamlit title
st.title("Market Basket Analysis for Cross-Selling and Upselling Opportunities")

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Dataset preview
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    # Dataset shape and missing values
    st.write(f"**Dataset size:** {df.shape}")
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())
    
    # Sidebar selection for analysis type
    analysis_type = st.sidebar.selectbox("Choose Analysis Type", ['Cross-Selling', 'Upselling'])

    if analysis_type == 'Upselling':
        # Add a 'Price' column (or use an existing one)
        # You can mock this for testing if you don’t have prices
        # For example purposes, let's generate random prices
        if 'Price' not in df.columns:
            np.random.seed(42)
            df['Price'] = np.random.randint(5, 100, size=len(df))
        
        # Convert 'Date' and 'Time' columns to a single 'Datetime' column
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df[["Datetime", "Transaction", "Item", "Price"]]
        
        # Filter out 'NONE' items
        df = df[df['Item'] != 'NONE']
    
    else:
        # For Cross-Selling, use only Transaction and Item columns
        # Convert 'Date' and 'Time' columns to a single 'Datetime' column
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df[["Datetime", "Transaction", "Item"]]
    
    # Common to both: Filter out 'NONE' items if any
    df = df[df['Item'] != 'NONE']
    
    # Calculate stats (shared for both cross-selling and upselling)
    total_items = len(df)
    total_days = len(np.unique(df.Datetime.dt.day))
    total_months = len(np.unique(df.Datetime.dt.month))
    average_items = int(total_items / total_days)
    unique_items = df.Item.unique().size
    
    # Display calculated stats
    st.write(f"**Total unique items sold:** {unique_items}")
    st.write(f"**Total sales:** {total_items} items sold in {total_days} days over {total_months} months")
    st.write(f"**Average daily sales:** {average_items} items")
    
    # Top 10 best-selling items
    counts = df.Item.value_counts()
    top_10 = counts[:10]
    
    st.write("**Top 10 Best-Selling Items:**")
    st.bar_chart(top_10)
    
    # Set Datetime as index for time-based analysis
    df.set_index('Datetime', inplace=True)
    
    # Time-based sales analysis
    if st.button("Show Daily Sales"):
        st.write("**Total Number of Items Sold by Date:**")
        df["Item"].resample("D").count().plot(figsize=(10,4))
        plt.title("Total Number of Items Sold by Date")
        plt.xlabel("Date")
        plt.ylabel("Total Number of Items Sold")
        st.pyplot(plt.gcf())

    if st.button("Show Monthly Sales"):
        st.write("**Total Number of Items Sold by Month:**")
        df["Item"].resample("M").count().plot(figsize=(10,4))
        plt.title("Total Number of Items Sold by Month")
        plt.xlabel("Date")
        plt.ylabel("Total Number of Items Sold")
        st.pyplot(plt.gcf())
    
    # Basket creation for Apriori
    df_basket = df.groupby(["Transaction","Item"]).size().reset_index(name="Count")
    market_basket = (df_basket.groupby(['Transaction', 'Item'])['Count'].sum().unstack().reset_index().fillna(0).set_index('Transaction'))
    
    # Map quantity to binary (1 if bought, 0 if not)
    market_basket = market_basket.applymap(lambda x: 1 if x > 0 else 0)
    
    # Apriori algorithm setup with user input for min_support
    st.sidebar.subheader("Apriori Algorithm Parameters")
    min_support = st.sidebar.slider("Minimum Support", min_value=0.01, max_value=0.5, value=0.02, step=0.01)

    if st.sidebar.button("Run Apriori"):
        itemsets = apriori(market_basket, min_support=min_support, use_colnames=True)
        st.write(f"**Frequent Itemsets with Min Support {min_support}:**")
        st.dataframe(itemsets.head())
        
        # Association rule mining
        metric = st.sidebar.selectbox("Metric for Association Rules", ['lift', 'confidence'])
        threshold = st.sidebar.slider(f"Minimum {metric.capitalize()}", min_value=0.1, max_value=1.0, value=0.5)
        
        rules = association_rules(itemsets, metric=metric, min_threshold=threshold)
        
        if analysis_type == 'Upselling':
            # Merge the 'Price' column with the association rules to focus on upselling
            item_prices = df[['Item', 'Price']].drop_duplicates()
            rules['antecedent_price'] = rules['antecedents'].apply(lambda x: item_prices[item_prices['Item'].isin(list(x))]['Price'].mean())
            rules['consequent_price'] = rules['consequents'].apply(lambda x: item_prices[item_prices['Item'].isin(list(x))]['Price'].mean())
            
            # Filter for upselling: where the consequent price is higher than the antecedent price
            upsell_rules = rules[rules['consequent_price'] > rules['antecedent_price']]
            
            st.write("**Upselling Association Rules:**")
            st.dataframe(upsell_rules[['antecedents', 'consequents', 'antecedent_price', 'consequent_price', 'support', 'confidence', 'lift']].head(20))
            
            # Scatter plot for support vs confidence
            support = upsell_rules['support'].values
            confidence = upsell_rules['confidence'].values
            
            plt.figure(figsize=(8,6))
            plt.scatter(support, confidence, alpha=0.5)
            plt.title('Upselling Rules: Support vs Confidence')
            plt.xlabel('Support')
            plt.ylabel('Confidence')
            st.pyplot(plt.gcf())
        
        else:
            # Cross-Selling: Simply display association rules without price focus
            st.write("**Cross-Selling Association Rules:**")
            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20))
            
            # Scatter plot for support vs confidence
            support = rules['support'].values
            confidence = rules['confidence'].values
            
            plt.figure(figsize=(8,6))
            plt.scatter(support, confidence, alpha=0.5)
            plt.title('Cross-Selling Rules: Support vs Confidence')
            plt.xlabel('Support')
            plt.ylabel('Confidence')
            st.pyplot(plt.gcf())
