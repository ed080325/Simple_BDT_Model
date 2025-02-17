import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

class BDT:
    def __init__(self, input_prices: dict[float, float], sigma: float):
        self.maturities = np.array(list(input_prices.keys()))
        self.prices = np.array(list(input_prices.values()))
        self.thetas = np.zeros(len(self.maturities) - 1)
        self.delta = self.maturities[0]
        self.z0 = math.log(-math.log(self.prices[0]/100) / self.delta)
        self.sigma = sigma

        try:
            self.solved_thetas = self.solve_thetas()
            self.plot_solved_tree(self.solved_thetas)
        except Exception as e:
            raise Exception(f"Error when solving model: {e}")

    def build_bdt_tree(self, thetas):
        n = len(self.maturities)
        
        # Initialize trees
        z_tree = [[0] * (i + 1) for i in range(n)]
        r_tree = [[0] * (i + 1) for i in range(n)]

        z_tree[0][0] = self.z0  # Set roots of the trees
        r_tree[0][0] = np.exp(z_tree[0][0])

        # Build tree iteratively
        for i in range(n - 1):
            for j in range(i + 1):
                up_move = z_tree[i][j] + thetas[i] * self.delta + self.sigma * np.sqrt(self.delta) # Define up and down moves separately to avoid calculation errors
                down_move = z_tree[i][j] + thetas[i] * self.delta - self.sigma * np.sqrt(self.delta)
                z_tree[i + 1][j] = up_move  # Only define z[i+1][j] as an up move from z[i][j] and not a down move from z[i][j-1]
                z_tree[i + 1][j + 1] = down_move

            for j in range(i + 2):
                r_tree[i + 1][j] = np.exp(z_tree[i + 1][j])

        return r_tree  # We only need the r tree so no need to store the z tree

    def bond_price_at_0(self, maturity, r_tree):
        price_tree = [[100] * (i + 1) for i in range(maturity + 1)] # Initialises all prices as 100

        # Work backwards to find price at time 0
        for i in range(maturity - 1, -1, -1):
            for j in range(i + 1):
                price_tree[i][j] = math.exp(-r_tree[i][j] * self.delta) * 0.5 * (price_tree[i + 1][j] + price_tree[i + 1][j + 1])  # Discounting formula

        return price_tree[0][0]  # Return price at time 0

    def objective(self, theta_i, i):
        self.thetas[i] = theta_i  # Set objective variable based on the theta we want to solve for
        r_tree = self.build_bdt_tree(self.thetas)  # Enable solver to recursively build new trees with updated guess
        p_guess = self.bond_price_at_0(i + 2, r_tree)  # Bond price at maturity i+2 since thetas[i] determines price of the bond maturing at [i+2]
        return p_guess - self.prices[i + 1]  # Difference from the actual price is the objective (thetas[i] determines p[i+1] which is the bond maturing at [i+2], since we take the first price as p0)

    def solve_thetas(self):
        for i in range(len(self.thetas)):
            sol = root_scalar(self.objective, args=(i,), bracket=[-1, 1], method='brentq')
            if sol.converged:
                self.thetas[i] = sol.root  # Dynamically update thetas as solved values are found
            else:
                raise ValueError(f"Root finding failed for thetas[{i}]")

        return self.thetas

    def plot_solved_tree(self, thetas):
        r_tree = self.build_bdt_tree(thetas)
        max_depth = len(r_tree)
        theta_labels = [f"{(theta*100):.2f}" for theta in thetas] + [""]  # Multiply thetas by 100 for readability

        max_rows = max(len(arr) for arr in r_tree)  # Find the max number of rows (longest array in r_tree)
        # Present r_tree in a table for easier visualisation
        table_data = []
        m_row = ["0"]
        i_row = []
        j_row = []
        for i in range(max_rows):
            i_row.append(i)
            j_row.append("")
        for i in range(max_rows - 1):
            m_row.append(self.maturities[i])
        table_data.append(m_row)
        table_data.append(theta_labels)  
        table_data.append(i_row)
        table_data.append(j_row)
        
        for i in range(max_rows):
            row = []
            for j in range(max_depth):
                if i < len(r_tree[j]):
                    row.append(f"{(r_tree[j][i]*100):.2f}%")  # Express rates as percentages
                else:
                    row.append("")
            table_data.append(row)

        fig, ax = plt.subplots(figsize=(max_depth, (max_rows + 1) * 0.5))

        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=table_data,
                         rowLabels=["Maturity"] + ["θ (x 100)"] + ["i"] + ["j"] + [f"{i}" for i in range(max_rows)],
                         loc="center",
                         cellLoc="center")

        for key, cell in table.get_celld().items():
            cell.set_linewidth(0)  # Remove borders for a clean look
        for j in range(max_depth):
            table[(1, j)].set_text_props(weight='bold')  # Make the theta row bold for clarity

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(max_depth)))

        plt.show()

if __name__ == "__main__":
    input_prices = {0.5 : 99.1338,
                    1.0 : 97.8925,
                    1.5 : 96.1462,
                    2.0 : 94.1011,
                    2.5 : 91.7136,
                    3.0 : 89.2258,
                    3.5 : 86.8142,
                    4.0 : 84.5016,
                    4.5 : 82.1848,
                    5.0 : 79.7718,
                    5.5 : 77.4339}
    sigma = 0.2142
    bdt_model = BDT(input_prices, sigma)
    solved_thetas = bdt_model.solve_thetas()
    solved_r_tree = bdt_model.build_bdt_tree(solved_thetas)
