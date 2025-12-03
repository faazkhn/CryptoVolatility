import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (15, 12)

def load_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'], thousands=',')
        df.columns = [c.strip().title() for c in df.columns]
        
        if 'Close/Last' not in df.columns:
            raise ValueError(f"Could not find 'Close/Last' column. Available: {df.columns}")

        df.rename(columns={'Close/Last': 'Price'}, inplace=True)
        
        if df['Price'].dtype == object:
            df['Price'] = df['Price'].astype(str).str.replace('$', '').str.replace(',', '')

        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        return df.dropna(subset=['Price'])
        
    except Exception as e:
        print(f"[Error] Failed to load File: {e}")
        return None

def calculate_metrics(df, risk_free_rate=0.04):
    if df.empty:
        return {
            'Total Return': 0.0,
            'Volatility': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown': 0.0
        }

    df['Returns'] = df['Price'].pct_change()
    
    try:
        total_return = (df['Price'].iloc[-1] / df['Price'].iloc[0]) - 1
    except IndexError:
        total_return = 0.0
    
    ann_volatility = df['Returns'].std() * np.sqrt(365)
    
    daily_rf = risk_free_rate / 365
    mean_daily_return = df['Returns'].mean()
    daily_std = df['Returns'].std()
    
    if daily_std == 0 or np.isnan(daily_std):
        sharpe_ratio = 0
    else:
        sharpe_ratio = (mean_daily_return - daily_rf) / daily_std * np.sqrt(365)
    
    rolling_max = df['Price'].cummax()
    drawdown = (df['Price'] / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    return {
        'Total Return': total_return,
        'Volatility': ann_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

def plot_performance(coins_dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    for name, df in coins_dict.items():
        if df.empty: continue
        
        norm_price = df['Price'] / df['Price'].iloc[0]
        ax1.plot(df.index, norm_price, label=name)
        
        rolling_max = df['Price'].cummax()
        drawdown = (df['Price'] / rolling_max) - 1
        ax2.plot(df.index, drawdown, label=name, alpha=0.7)

    ax1.set_title('Growth of $1 Investment (Normalized)', fontsize=14)
    ax1.set_ylabel('Multiplier')
    ax1.legend()
    
    ax2.set_title('Drawdown (Decline from Peak)', fontsize=14)
    ax2.set_ylabel('Drawdown %')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_correlation(coins_dict):
    if len(coins_dict) < 2:
        print("[Info] Need at least 2 coins for correlation analysis.")
        return

    combined_returns = pd.DataFrame()
    for name, df in coins_dict.items():
        if not df.empty:
            combined_returns[name] = df['Returns']
    
    if combined_returns.empty:
        print("[Info] No valid data for correlation.")
        return

    plt.figure(figsize=(10, 8))
    corr = combined_returns.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Correlation Matrix (Daily Returns)', fontsize=16)
    plt.show()

def main_menu():
    coins = {}
    
    while True:
        print("\n" + "="*40)
        print("   CRYPTO ANALYZER")
        print("="*40)
        print(f"Loaded Coins: {list(coins.keys()) if coins else 'None'}")
        print("-" * 40)
        print("1. Load New Coin (CSV)")
        print("2. View Risk/Reward Report")
        print("3. Plot Performance Charts")
        print("4. Plot Correlation Heatmap")
        print("5. Clear All Data")
        print("6. Exit")
        
        choice = input("\nSelect Option (1-6): ").strip()
        
        if choice == '1':
            ticker = input("Enter Coin Symbol (e.g., BTC): ").upper().strip()
            path = input("Enter CSV Path: ").strip().replace('"', '').replace("'", "")
            
            if os.path.exists(path):
                df = load_data_from_csv(path)
                if df is not None and not df.empty:
                    df['Returns'] = df['Price'].pct_change()
                    coins[ticker] = df
                    print(f"[Success] {ticker} loaded with {len(df)} rows.")
                else:
                    print("[Error] File loaded but contained no valid price data.")
            else:
                print("[Error] File not found.")
                
        elif choice == '2':
            if not coins:
                print("[!] Please load data first.")
                continue
            
            print("\n" + "-"*85)
            print(f"{'COIN':<10} | {'RETURN':<10} | {'VOLATILITY':<12} | {'SHARPE RATIO':<15} | {'MAX DRAWDOWN':<15}")
            print("-" * 85)
            
            for name, df in coins.items():
                m = calculate_metrics(df)
                print(f"{name:<10} | {m['Total Return']:>9.2%} | {m['Volatility']:>11.2%} | {m['Sharpe Ratio']:>14.2f} | {m['Max Drawdown']:>14.2%}")
            print("-" * 85)
            input("\nPress Enter to continue...")

        elif choice == '3':
            if coins: plot_performance(coins)
            else: print("[!] No data loaded.")

        elif choice == '4':
            if coins: plot_correlation(coins)
            else: print("[!] No data loaded.")
            
        elif choice == '5':
            coins = {}
            print("[Info] All data cleared.")
            
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("[!] Invalid option.")

main_menu()