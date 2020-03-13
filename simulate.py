import argparse
import pandas as pd
import numpy as np
import wandb
import os.path as path
import math
from hyperopt import hp, fmin, tpe


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
    parser.add_argument('--coefficients', help='Coefficients',
                        type=float, nargs=3)
    parser.add_argument('--max_evals', help='Max Evals for param search',
                        type=int)
    parser.add_argument('--c1_init', help='Mean and STD for C1',
                        type=float, nargs=2)
    parser.add_argument('--c2_init', help='Mean and STD for C2',
                        type=float, nargs=2)
    parser.add_argument('--c3_init', help='Mean and STD for C3',
                        type=float, nargs=2)

    args = parser.parse_args()

    return args


def objective(args, evaluation=False, log_run=True):
    # print('c1: {}\tc2: {}\tc3: {}'.format(args['c1'], args['c2'], args['c3']))
    if args['c2'] <= 0 or args['c3'] <= 0:
        return 1e10

    btc_usd_frame = args['btc_usd_frame']
    results_df = pd.DataFrame(columns=[
                              'Run', 'Current BTC', 'Total Profit', 'Average BTC Price', 'Ending BTC Price'])
    for _ in range(args['num_runs']):

        start_index = np.random.randint(
            0, len(btc_usd_frame) - args['num_weeks'] * 7)
        end_index = start_index + args['num_weeks'] * 7

        sample_frame = btc_usd_frame[start_index:end_index]
        current_price = args['current_price']
        total_profit = 0.0

        current_btc_amount = args['total_btc']
        weeks = 0
        average_btc_price = 0
        cnt = 0
        for _, row in sample_frame.iterrows():
            cnt += 1
            percent_change = float(row['Perc Close'].strip('%')) / 100
            current_price = current_price + current_price * percent_change
            average_btc_price += current_price

            if cnt % 7 != 0 or current_price <= 2500:
                continue
            weeks += 1
            # price_to_sell = args['c1'] * current_price ** 2 + \
            #     args['c2'] * current_price + args['c3']

            btc_to_sell = args['c1'] + args['c2'] * \
                math.log(args['c3'] * (current_price - 2500), 10)
            btc_to_sell = min(btc_to_sell, current_btc_amount)
            # print('btc_to_sell: {}'.format(btc_to_sell))

            # btc_to_sell = price_to_sell / current_price

            # if current_btc_amount < btc_to_sell or price_to_sell <= 0:
            if btc_to_sell <= 0:
                continue

            current_btc_amount = current_btc_amount - btc_to_sell
            total_profit = total_profit + btc_to_sell * current_price

            if args['num_runs'] == 1:
                print('\t', _)
                print('\tPercent Change: {:.4f}'.format(percent_change))
                print('\tCurrent price: {:.2f}'.format(current_price))
                print('\tAverage price: {:.2f}'.format(
                    average_btc_price / cnt))
                # print('\t\tPrice to sell: {:.2f}'.format(price_to_sell))
                print(
                    '\t\tBTC to sell: {:.4f} / {:.4f}'.format(btc_to_sell, current_btc_amount))
                print('\t\tTotal profit: {:.2f}'.format(total_profit))

        average_btc_price /= cnt
        results_df = results_df.append({'Current BTC': current_btc_amount,
                                        'Total Profit': total_profit, 'Ending BTC Price': current_price, 'Average BTC Price': average_btc_price}, ignore_index=True)

    if results_df['Total Profit'].min() < 1000 or results_df['Total Profit'].std() < 1000 or results_df['Total Profit'].max() < 1000:
        score = 1e10
    else:
        remaining_btc_ratio = args['total_btc'] / (current_btc_amount + 1e-10)
        remaining_btc_ratio = remaining_btc_ratio / 10

        score = remaining_btc_ratio + \
            results_df['Total Profit'].std() / results_df['Total Profit'].min()

    summary = {
        'Mean': results_df['Total Profit'].mean(),
        'STD': results_df['Total Profit'].std(),
        'Median': results_df['Total Profit'].median(),
        'Min': results_df['Total Profit'].min(),
        'Max': results_df['Total Profit'].max(),
        'Range': results_df['Total Profit'].max() - results_df['Total Profit'].min(),
        'Average BTC Price': average_btc_price,
        'Ending BTC Price': current_price,
        'Current BTC': current_btc_amount,
        'score': score,
        'c1': args['c1'],
        'c2': args['c2'],
        'c3': args['c3']
    }

    if evaluation == True:
        results_df.to_csv('results.csv')
        print(summary)
        # wandb.run.summary = summary
        # wandb.run.summary.update()
        # print(results_df)
        print('Saved to: results.csv')

    if log_run == True:
        summary_df = pd.DataFrame()
        summary_df = summary_df.append(summary, ignore_index=True)
        wandb.log(summary)

        with open(path.join(wandb.run.dir, 'coefficients.csv'), 'a') as f:
            summary_df.to_csv(f, header=f.tell() == 0)

    return score


def get_space(args, btc_usd_frame, init_c1, init_c2, init_c3):
    if args.coefficients:
        c1 = args.coefficients[0]
        c2 = args.coefficients[1]
        c3 = args.coefficients[2]
    else:
        c1 = hp.normal('c1', init_c1['mean'], init_c1['sigma'])
        c2 = hp.lognormal('c2', init_c2['mean'], init_c2['sigma'])
        c3 = hp.lognormal('c3', init_c3['mean'], init_c3['sigma'])

    return {
        'c1': c1,
        'c2': c2,
        'c3': c3,
        'current_price': args.current_price,
        'total_btc': args.total_btc,
        'num_runs': args.num_runs,
        'num_weeks': args.num_weeks,
        'btc_usd_frame': btc_usd_frame
    }


def main():
    args = parse_args()
    print('Starting.... {}\n'.format(args.current_price))
    evaluation = False

    if args.coefficients:
        evaluation = True
        log_run = False

    btc_usd_frame = pd.read_csv('BTC-USD.csv')

    # init_c1 = {'mean': 2, 'sigma': 0.1}
    # init_c2 = {'mean': 1.5, 'sigma': 0.1}
    # init_c3 = {'mean': 0.00001, 'sigma': 0.0001}
    # init_c1 = {'mean': -2, 'sigma': 0.1}
    # init_c2 = {'mean': 1, 'sigma': 0.1}
    # init_c3 = {'mean': 0.2, 'sigma': 0.0001}
    # init_c1 = {'mean': -2.86151, 'sigma': 2.52158}
    # init_c2 = {'mean': 0.96487, 'sigma': 0.66938}
    # init_c3 = {'mean': 2.6609572, 'sigma': 1.50999}
    init_c1 = {'mean': args.c1_init[0], 'sigma': args.c1_init[1]}
    init_c2 = {'mean': args.c2_init[0], 'sigma': args.c2_init[1]}
    init_c3 = {'mean': args.c3_init[0], 'sigma': args.c3_init[1]}

    space = get_space(args, btc_usd_frame, init_c1, init_c2, init_c3)

    if evaluation == False:
        wandb.init(project="btc-sell", config={
            "Max Eval": args.max_evals,
            "Num Runs": args.num_runs,
            "Total BTC": args.total_btc,
            "Current Price": args.current_price,
            'C1_mean': init_c1['mean'],
            'C1_sigma': init_c1['sigma'],
            'C2_mean': init_c2['mean'],
            'C2_sigma': init_c2['sigma'],
            'C3_mean': init_c3['mean'],
            'C3_sigma': init_c3['sigma']
        })

        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=args.max_evals)
        print('Best Summary: ', best)
        space['c1'] = best['c1']
        space['c2'] = best['c2']
        space['c3'] = best['c3']
        log_run = True

    eval_score = objective(space, evaluation=True, log_run=log_run)
    print('Evaluation Score: ', eval_score)


if __name__ == '__main__':
    main()
