class AnomalyGate:

    def __init__(self):
        pass

    def update_state(self, dt, measurement, prediction, label):
        """
        Decision logic for updating the Digital Twin trusted state.

        label = 0 → CLEAN → trust sensor
        label = 1 → ANOMALY → trust DT prediction
        """

        # ------------------------------
        # CLEAN sensor → trust sensor
        # ------------------------------
        if label == 0:

            print("\nDecision: CLEAN → using SENSOR measurement")

            trusted_T  = measurement["T"]
            trusted_Td = measurement["Td"]
            trusted_RH = measurement["RH"]

        # ------------------------------
        # ANOMALY → trust DT prediction
        # ------------------------------
        else:

            print("\nDecision: ANOMALY → replacing with DT prediction")

            trusted_T  = prediction["T_pred"]
            trusted_Td = prediction["Td_pred"]
            trusted_RH = prediction["RH_pred"]

        # ------------------------------
        # Update Digital Twin state
        # ------------------------------
        dt.update_state(
            trusted_T,
            trusted_Td,
            trusted_RH
        )
