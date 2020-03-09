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

    print('Starting....\n')
    # coefficient_1 = np.arange(1e-5, 1e-3, 1e-5)
    # coefficient_2 = np.arange(1e-5, 1e-3, 1e-5)
    # coefficient_3 = np.arange(0, 1000, 100)
    coefficient_1 = np.linspace(0, 1e-3, num=10, dtype=np.float16)
    coefficient_2 = np.linspace(0, 1e-3, num=10, dtype=np.float16)
    coefficient_3 = np.linspace(0, 1000, num=10, dtype=np.float16)
    coefficients = []
    for c1 in coefficient_1:
        for c2 in coefficient_2:
            for c3 in coefficient_3:
                coefficients.append([c1, c2, c3])
    np.random.shuffle(coefficients)

    best_avg_total_profit = 0
    best_coefficients = []
    c_index = 0
    for coefficient in coefficients:
        c_index += 1
        results_df = pd.DataFrame(
            columns=['Run', 'Current BTC', 'Total Profit'])
        for run in range(args.num_runs):

            start_index = np.random.randint(
                0, len(btc_usd_frame) - args.num_weeks * 7)
            end_index = start_index + args.num_weeks * 7

            sample_frame = btc_usd_frame[start_index:end_index]
            current_price = args.current_price
            total_profit = 0.0

            current_btc_amount = args.total_btc
            weeks = 0
            average_btc_price = 0
            for index, row in sample_frame.iterrows():
                average_btc_price += current_price
                percent_change = float(row['Perc Close'].strip('%')) / 100
                current_price = current_price + current_price * percent_change

                if index % 7 != 0:
                    continue
                weeks += 1
                price_to_sell = coefficient[0] * current_price ** 2 + \
                    coefficient[1] * current_price + coefficient[2]

                # if current_price >= 5000 and current_price < 10000:
                #     price_to_sell = 1000
                # elif current_price >= 10000 and current_price < 20000:
                #     price_to_sell = 2500
                # elif current_price >= 20000 and current_price < 30000:
                #     price_to_sell = 5000
                # elif current_price >= 30000:
                #     price_to_sell = 10000
                # elif current_price >= 40000:
                #     price_to_sell = 20000

                btc_to_sell = price_to_sell / current_price

                if current_btc_amount <= btc_to_sell:
                    continue

                current_btc_amount = current_btc_amount - btc_to_sell
                total_profit = total_profit + btc_to_sell * current_price

                # print('Price: {:.2f}\tCurrent BTC Amount: {:.2f}\tTotal Profit: {:.2f}\tSelling {:.2f} at {:.2f}'.format(
                #     current_price, current_btc_amount, total_profit,  btc_to_sell, price_to_sell))

            # print('weeks', weeks)
            average_btc_price /= len(sample_frame)
            results_df = results_df.append({'Run': run, 'Current BTC': current_btc_amount,
                                            'Total Profit': total_profit, 'Ending BTC Price': current_price, 'Average BTC Price': average_btc_price}, ignore_index=True)

        print(c_index, '/', len(coefficients))
        summary_df = pd.DataFrame(
            columns=['Mean', 'STD', 'Median', 'Min', 'Max', 'Range'])
        current_btc_row = pd.Series({
            'Mean': results_df['Current BTC'].mean(),
            'STD': results_df['Current BTC'].std(),
            'Median': results_df['Current BTC'].median(),
            'Min': results_df['Current BTC'].min(),
            'Max': results_df['Current BTC'].max(),
            'Range': results_df['Current BTC'].max() - results_df['Current BTC'].min(),

        }, name='Current BTC')

        total_profit_row = pd.Series({
            'Mean': results_df['Total Profit'].mean(),
            'STD': results_df['Total Profit'].std(),
            'Median': results_df['Total Profit'].median(),
            'Min': results_df['Total Profit'].min(),
            'Max': results_df['Total Profit'].max(),
            'Range': results_df['Total Profit'].max() - results_df['Total Profit'].min(),

        }, name='Total Profit')

        ending_btc_price_row = pd.Series({
            'Mean': results_df['Ending BTC Price'].mean(),
            'STD': results_df['Ending BTC Price'].std(),
            'Median': results_df['Ending BTC Price'].median(),
            'Min': results_df['Ending BTC Price'].min(),
            'Max': results_df['Ending BTC Price'].max(),
            'Range': results_df['Ending BTC Price'].max() - results_df['Ending BTC Price'].min(),

        }, name='Ending BTC Price')

        average_btc_price_row = pd.Series({
            'Mean': results_df['Average BTC Price'].mean(),
            'STD': results_df['Average BTC Price'].std(),
            'Median': results_df['Average BTC Price'].median(),
            'Min': results_df['Average BTC Price'].min(),
            'Max': results_df['Average BTC Price'].max(),
            'Range': results_df['Average BTC Price'].max() - results_df['Average BTC Price'].min(),

        }, name='BTC Price')

        summary_df = summary_df.append(
            current_btc_row).append(total_profit_row).append(ending_btc_price_row).append(average_btc_price_row)
        pd.options.display.float_format = '{:,}'.format
        print(summary_df.round(2))

        print('\t', coefficient)

        if best_avg_total_profit < results_df['Total Profit'].mean() and results_df['Total Profit'].std() <= results_df['Total Profit'].mean() / 2:
            best_avg_total_profit = results_df['Total Profit'].mean()
            best_coefficients = coefficient
            print('\tBest Average Total Profit: {:.2f}'.format(
                best_avg_total_profit))
            open('coefficients.out', 'w').write(
                str(best_coefficients) + '\n')
            results_df.to_csv('best_results.csv')

        results_df.to_csv('results.csv')


if __name__ == '__main__':
    main()
