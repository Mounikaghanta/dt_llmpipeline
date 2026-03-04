class AnomalyGate:
    """
    Decision module that controls Digital Twin state assimilation.

    If classifier label == 0 (clean):
        update DT with real sensor measurement.

    If classifier label == 1 (anomaly):
        update DT with DT's own predicted values.
    """

    def update_state(self, dt, sensor_row: dict, prediction: dict, anomaly_label: int):
        """
        Parameters
        dt : DigitalTwinV2
            The digital twin instance.
        sensor_row : dict
            {"T": float, "Td": float, "RH": float}
        prediction : dict
            {"T_pred": float, "Td_pred": float, "RH_pred": float}
        anomaly_label : int
            0 = clean, 1 = anomaly
        """

        if anomaly_label == 0:
            # Clean → trust measurement
            dt.update_state(
                sensor_row["T"],
                sensor_row["Td"],
                sensor_row["RH"]
            )
        else:
            # Anomaly → trust prediction
            dt.update_state(
                prediction["T_pred"],
                prediction["Td_pred"],
                prediction["RH_pred"]
            )
