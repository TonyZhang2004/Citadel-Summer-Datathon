import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_health_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    
    df.dropna(subset=['Data_Value', 'YearStart', 'Sample_Size'], inplace=True)
    
    df['YearStart'] = df['YearStart'].astype(int)
    df['Data_Value'] = pd.to_numeric(df['Data_Value'], errors='coerce')
    df['Sample_Size'] = pd.to_numeric(df['Sample_Size'], errors='coerce')
    
    df = df[['YearStart', 'Topic', 'Data_Value', 'Sample_Size']]
    
    return df

def weighted_mean(df, value_col, weight_col):
    d = df[value_col]
    w = df[weight_col]
    return (d * w).sum() / w.sum()

def compute_yearly_weighted_averages(df, topics):
    yearly_data = pd.DataFrame()
    for topic in topics:
        topic_df = df[df['Topic'] == topic]
        yearly_topic = topic_df.groupby('YearStart').apply(lambda x: pd.Series({
            f'{topic}_Weighted_Avg': weighted_mean(x, 'Data_Value', 'Sample_Size')
        })).reset_index()
        
        if yearly_data.empty:
            yearly_data = yearly_topic
        else:
            yearly_data = pd.merge(yearly_data, yearly_topic, on='YearStart', how='outer')
    
    yearly_data.rename(columns={'YearStart': 'Year'}, inplace=True)
    return yearly_data

def load_and_clean_stock_data(file_path):
    df = pd.read_csv(file_path)
    
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    
    df['Year'] = df['Date-Time'].dt.year
    
    df = df[['Year', 'Ticker_Symbol', 'Close']]
    
    return df

def compute_yearly_stock_averages(df):
    return df.groupby(['Year', 'Ticker_Symbol']).mean().reset_index()

def merge_datasets(health_df, stock_df):
    stock_df_pivot = stock_df.pivot(index='Year', columns='Ticker_Symbol', values='Close').reset_index()
    merged_data = pd.merge(health_df, stock_df_pivot, on='Year', how='inner')
    return merged_data

def plt_corr_matrix(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Year'])    
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(14, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Various Health Metrics and Stock Prices')
    plt.show()
    
    return corr_matrix

def identify_high_correlations(corr_matrix, threshold=0.5):
    health_metrics = [col for col in corr_matrix.columns if 'Weighted_Avg' in col]
    stock_symbols = [col for col in corr_matrix.columns if col not in health_metrics]
    
    high_corrs = []
    
    for metric in health_metrics:
        for stock in stock_symbols:
            corr_value = corr_matrix.at[metric, stock]
            if abs(corr_value) >= threshold:
                high_corrs.append((stock, metric, corr_value))
    
    high_corrs = sorted(high_corrs, key=lambda x: abs(x[2]), reverse=True)
    
    return high_corrs

def main():
    health_file_path = 'Nutrition_Physical_Activity_and_Obesity_Data.csv'  # Replace with the path to your health CSV file
    stock_file_path = 'all_stock_and_etfs.csv'  # Replace with the path to your stock/ETF CSV file
    
    health_df = load_and_clean_health_data(health_file_path)
    stock_df = load_and_clean_stock_data(stock_file_path)
    stock_df = compute_yearly_stock_averages(stock_df)
    
    topics = ['Sugar Drinks - Behavior', 'Obesity / Weight Status', 'Fruits and Vegetables - Behavior', 'Physical Activity - Behavior']  # Replace with the topics you want to analyze
    
    health_yearly_avg = compute_yearly_weighted_averages(health_df, topics)
    
    merged_data = merge_datasets(health_yearly_avg, stock_df)
    
    print("Merged Data:")
    print(merged_data.head())
    
    corr_matrix = plt_corr_matrix(merged_data)
    
    high_corrs = identify_high_correlations(corr_matrix, threshold=0.5) 
    
    print("High Correlations:")
    for stock, metric, corr_value in high_corrs:
        print(f"Stock: {stock}, Metric: {metric}, Correlation: {corr_value:.2f}")

if __name__ == "__main__":
    main()
