import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--total_btc', help='Total amount of BTC to sell', type=float,
                        default=1)
    parser.add_argument('--num_runs', help='Number of runs',
                        type=int, default=100)
    parser.add_argument('--num_weeks', help='Number of weeks',
                        type=int, default=52)
    parser.add_argument('--current_price', help='Current price of BTC',
                        type=float, required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    btc_usd_frame = pd.read_csv('BTC-USD.csv')
    results_df = pd.DataFrame(columns=['Run', 'Current BTC', 'Total Profit'])

    print('Starting....\n')

    for run in range(args.num_runs):

        start_index = np.random.randint(
            0, len(btc_usd_frame) - args.num_weeks * 7)
        end_index = start_index + args.num_weeks * 7

        sample_frame = btc_usd_frame[start_index:end_index]
        # print(start_index, end_index, len(sample_frame))
        current_price = args.current_price
        total_profit = 0.0

        current_btc_amount = args.total_btc
        weeks = 0
        for index, row in sample_frame.iterrows():
            percent_change = float(row['Perc Close'].strip('%')) / 100
            current_price = current_price + current_price * percent_change

            if index % 7 != 0:
                continue
            weeks += 1
            price_to_sell = 1e-5 * current_price ** 2 + 0.0008 * current_price + 373
            # price_to_sell = 0.1 * current_price

            if current_price >= 5000 and current_price < 10000:
                price_to_sell = 1000
            elif current_price >= 10000 and current_price < 20000:
                price_to_sell = 2500
            elif current_price >= 20000 and current_price < 30000:
                price_to_sell = 5000
            elif current_price >= 30000:
                price_to_sell = 10000
            elif current_price >= 40000:
                price_to_sell = 20000

            btc_to_sell = price_to_sell / current_price

            if current_btc_amount <= btc_to_sell:
                continue

            current_btc_amount = current_btc_amount - btc_to_sell
            total_profit = total_profit + btc_to_sell * current_price

            # print('Price: {:.2f}\tCurrent BTC Amount: {:.2f}\tTotal Profit: {:.2f}\tSelling {:.2f} at {:.2f}'.format(
            #     current_price, current_btc_amount, total_profit,  btc_to_sell, price_to_sell))

        # print('weeks', weeks)
        results_df = results_df.append({'Run': run, 'Current BTC': current_btc_amount,
                                        'Total Profit': total_profit, 'Ending BTC Price': current_price}, ignore_index=True)

    print('Average BTC Left: {:.2f} ± {:.2f}\tAverage Total Profit: {:.2f} ± {:.2f}\t Average Ending BTC Price: {:.2f} ± {:.2f}\t'.format(
        results_df['Current BTC'].mean(), results_df['Current BTC'].std(), results_df['Total Profit'].mean(), results_df['Total Profit'].std(), results_df['Ending BTC Price'].mean(), results_df['Ending BTC Price'].std()))

    results_df.to_csv('results.csv')


if __name__ == '__main__':
    main()
