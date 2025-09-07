import pandas as pd
import matplotlib.pyplot as plt
import scienceplots


# Read both CSV files
oc_inr_ot_data = pd.read_csv('OC_inr_ot_loss.csv')
oc_inr_data = pd.read_csv('OC_inr_data_loss.csv')
inr_data = pd.read_csv('inr_data_loss.csv')

# Figure 1: Data Loss
with plt.style.context('science'):
    plt.figure()
    plt.plot(
        oc_inr_data["trainer/global_step"],
        oc_inr_data["pious-vortex-640 - train/data_loss"],
        label="Data Loss"
    )
    plt.xlabel("Global Step")
    plt.ylabel("Train Data Loss")
    plt.title("OC INR Train Data Loss Visualization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("oc_inr_data_loss_plot.png", dpi=300)
    plt.close()

# Figure 2: OT Loss
with plt.style.context('science'):
    plt.figure()
    plt.plot(
        oc_inr_ot_data["trainer/global_step"],
        oc_inr_ot_data["pious-vortex-640 - train/ot_loss"],
        label="OT Loss"
    )
    plt.xlabel("Global Step")
    plt.ylabel("Train OT Loss")
    plt.title("OC INR Train OT Loss Visualization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("oc_inr_ot_loss_plot.png", dpi=300)
    plt.close()

# Figure 3: Data Loss
with plt.style.context('science'):
    plt.figure()
    plt.plot(
        inr_data["trainer/global_step"],
        inr_data["frosty-sky-643 - train/loss"],
        label="Data Loss"
    )
    plt.xlabel("Global Step")
    plt.ylabel("Train Data Loss")
    plt.title("INR Train Data Loss Visualization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inr_data_loss_plot.png", dpi=300)
    plt.close()