PRESET_PARAMS = {
    "FedDeblur": {
        "TV": {
            "complete": {
                (10, 0.0): {
                    "grayscale": {"eta": 0.00005, "rho": 0.0003},
                    "color": {"eta": 0.00001, "rho": 0.00075},
                },
                (5, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.0001},
                    "color": {"eta": 0.0001, "rho": 0.0001},
                },
                (3, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.0001},
                    "color": {"eta": 0.0001, "rho": 0.0001},
                },
                (10, 0.5): {
                    "grayscale": {"eta": 0.0075, "rho": 0.0025},
                    "color": {"eta": 0.01, "rho": 0.002},
                },
                (5, 0.5): {
                    "grayscale": {"eta": 0.01, "rho": 0.0025},
                    "color": {"eta": 0.02, "rho": 0.005},
                },
                (3, 0.5): {
                    "grayscale": {"eta": 0.02, "rho": 0.0075},
                    "color": {"eta": 0.025, "rho": 0.0075},
                },
                (10, 1.0): {
                    "grayscale": {"eta": 0.025, "rho": 0.0005},
                    "color": {"eta": 0.035, "rho": 0.001},
                },
                (9, 1.0): {
                    "grayscale": {"eta": 0.025, "rho": 0.0025},
                    "color": {"eta": 0.03, "rho": 0.0025},
                },
                (8, 1.0): {
                    "grayscale": {"eta": 0.025, "rho": 0.003},
                    "color": {"eta": 0.03, "rho": 0.003},
                },
                (7, 1.0): {
                    "grayscale": {"eta": 0.03, "rho": 0.005},
                    "color": {"eta": 0.05, "rho": 0.003},
                },
                (6, 1.0): {
                    "grayscale": {"eta": 0.03, "rho": 0.003},
                    "color": {"eta": 0.05, "rho": 0.005},
                },
                (5, 1.0): {
                    "grayscale": {"eta": 0.035, "rho": 0.005},
                    "color": {"eta": 0.05, "rho": 0.005},
                },
                (4, 1.0): {
                    "grayscale": {"eta": 0.05, "rho": 0.03},
                    "color": {"eta": 0.075, "rho": 0.05},
                },
                (3, 1.0): {
                    "grayscale": {"eta": 0.05, "rho": 0.01},
                    "color": {"eta": 0.075, "rho": 0.02},
                },
                (2, 1.0): {
                    "grayscale": {"eta": 0.05, "rho": 0.0001},
                    "color": {"eta": 0.075, "rho": 0.0001},
                },
                (1, 1.0): {
                    "grayscale": {"eta": 0.1, "rho": 0.0001},
                    "color": {"eta": 0.2, "rho": 0.0001},
                },
                (10, 2.0): {
                    "grayscale": {"eta": 0.075, "rho": 0.001},
                    "color": {"eta": 0.1, "rho": 0.0015},
                },
                (5, 2.0): {
                    "grayscale": {"eta": 0.1, "rho": 0.0008},
                    "color": {"eta": 0.2, "rho": 0.002},
                },
                (3, 2.0): {
                    "grayscale": {"eta": 0.2, "rho": 0.03},
                    "color": {"eta": 0.2, "rho": 0.02},
                },
                (10, 3.0): {
                    "grayscale": {"eta": 0.1, "rho": 0.001},
                    "color": {"eta": 0.2, "rho": 0.002},
                },
                (5, 3.0): {
                    "grayscale": {"eta": 0.2, "rho": 0.002},
                    "color": {"eta": 0.3, "rho": 0.001},
                },
                (3, 3.0): {
                    "grayscale": {"eta": 0.5, "rho": 0.1},
                    "color": {"eta": 0.5, "rho": 0.1},
                },
            },
            "partial": {
                "motion": {
                    0.0: {
                        "grayscale": {"eta": 0.0005, "rho": 0.0001},
                        "color": {"eta": 0.00075, "rho": 0.0001},
                    },
                    0.5: {
                        "grayscale": {"eta": 0.005, "rho": 0.003},
                        "color": {"eta": 0.005, "rho": 0.0003},
                    },
                    1.0: {
                        "grayscale": {"eta": 0.015, "rho": 0.005},
                        "color": {"eta": 0.02, "rho": 0.005},
                    },
                    2.0: {
                        "grayscale": {"eta": 0.035, "rho": 0.0003},
                        "color": {"eta": 0.015, "rho": 0.000075},
                    },
                    3.0: {
                        "grayscale": {"eta": 0.05, "rho": 0.0002},
                        "color": {"eta": 0.008, "rho": 0.00002},
                    },
                },
                "synthesis": {
                    0.0: {
                        "grayscale": {"eta": 0.0003, "rho": 0.000075},
                        "color": {"eta": 0.0002, "rho": 0.00003},
                    },
                    0.5: {
                        "grayscale": {"eta": 0.003, "rho": 0.0005},
                        "color": {"eta": 0.005, "rho": 0.001},
                    },
                    1.0: {
                        "grayscale": {"eta": 0.01, "rho": 0.002},
                        "color": {"eta": 0.01, "rho": 0.0005},
                    },
                    2.0: {
                        "grayscale": {"eta": 0.035, "rho": 0.003},
                        "color": {"eta": 0.05, "rho": 0.003},
                    },
                    3.0: {
                        "grayscale": {"eta": 0.08, "rho": 0.025},
                        "color": {"eta": 0.08, "rho": 0.005},
                    },
                },
            },
        },
        "PnP": {
            "complete": {
                (3, 1.0): {"grayscale": {"sigma": 5.0, "rho": 0.0075}},
                (5, 1.0): {"grayscale": {"sigma": 10.0, "rho": 0.001}},
                (10, 1.0): {"grayscale": {"sigma": 5.0, "rho": 0.001}},
            },
            "partial": {
                "motion": {1.0: {"grayscale": {"sigma": 25.0, "rho": 0.02}}},
                "synthesis": {1.0: {"grayscale": {"sigma": 30.0, "rho": 0.015}}},
            },
        },
    },
    "CenDeblur": {
        "TV": {
            "complete": {
                (10, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.0003},
                    "color": {"eta": 0.0001, "rho": 0.0001},
                },
                (5, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.0001},
                    "color": {"eta": 0.0001, "rho": 0.0001},
                },
                (3, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.0001},
                    "color": {"eta": 0.0001, "rho": 0.0001},
                },
                (10, 0.5): {
                    "grayscale": {"eta": 0.008, "rho": 0.0001},
                    "color": {"eta": 0.01, "rho": 0.0001},
                },
                (5, 0.5): {
                    "grayscale": {"eta": 0.01, "rho": 0.0001},
                    "color": {"eta": 0.02, "rho": 0.0001},
                },
                (3, 0.5): {
                    "grayscale": {"eta": 0.02, "rho": 0.03},
                    "color": {"eta": 0.03, "rho": 0.03},
                },
                (10, 1.0): {
                    "grayscale": {"eta": 0.025, "rho": 0.0001},
                    "color": {"eta": 0.035, "rho": 0.0001},
                },
                (5, 1.0): {
                    "grayscale": {"eta": 0.035, "rho": 0.0001},
                    "color": {"eta": 0.05, "rho": 0.0001},
                },
                (3, 1.0): {
                    "grayscale": {"eta": 0.05, "rho": 0.0001},
                    "color": {"eta": 0.075, "rho": 0.0001},
                },
                (10, 2.0): {
                    "grayscale": {"eta": 0.08, "rho": 0.1},
                    "color": {"eta": 0.1, "rho": 0.05},
                },
                (5, 2.0): {
                    "grayscale": {"eta": 0.1, "rho": 0.0002},
                    "color": {"eta": 0.2, "rho": 0.1},
                },
                (3, 2.0): {
                    "grayscale": {"eta": 0.2, "rho": 0.001},
                    "color": {"eta": 0.2, "rho": 0.1},
                },
                (10, 3.0): {
                    "grayscale": {"eta": 0.15, "rho": 0.2},
                    "color": {"eta": 0.15, "rho": 0.1},
                },
                (5, 3.0): {
                    "grayscale": {"eta": 0.2, "rho": 0.2},
                    "color": {"eta": 0.3, "rho": 0.3},
                },
                (3, 3.0): {
                    "grayscale": {"eta": 0.3, "rho": 0.5},
                    "color": {"eta": 0.5, "rho": 0.5},
                },
            },
            "partial": {
                "motion": {
                    0.0: {
                        "grayscale": {"eta": 0.002, "rho": 0.02},
                        "color": {"eta": 0.003, "rho": 0.003},
                    },
                    0.5: {
                        "grayscale": {"eta": 0.005, "rho": 0.003},
                        "color": {"eta": 0.008, "rho": 0.002},
                    },
                    1.0: {
                        "grayscale": {"eta": 0.015, "rho": 0.005},
                        "color": {"eta": 0.02, "rho": 0.005},
                    },
                    2.0: {
                        "grayscale": {"eta": 0.05, "rho": 0.003},
                        "color": {"eta": 0.05, "rho": 0.0005},
                    },
                    3.0: {
                        "grayscale": {"eta": 0.08, "rho": 0.002},
                        "color": {"eta": 0.1, "rho": 0.002},
                    },
                },
                "synthesis": {
                    0.0: {
                        "grayscale": {"eta": 0.0005, "rho": 0.03},
                        "color": {"eta": 0.001, "rho": 0.03},
                    },
                    0.5: {
                        "grayscale": {"eta": 0.003, "rho": 0.01},
                        "color": {"eta": 0.005, "rho": 0.01},
                    },
                    1.0: {
                        "grayscale": {"eta": 0.01, "rho": 0.01},
                        "color": {"eta": 0.01, "rho": 0.001},
                    },
                    2.0: {
                        "grayscale": {"eta": 0.03, "rho": 0.0005},
                        "color": {"eta": 0.05, "rho": 0.05},
                    },
                    3.0: {
                        "grayscale": {"eta": 0.08, "rho": 0.2},
                        "color": {"eta": 0.08, "rho": 0.0005},
                    },
                },
            },
        },
        "PnP": {
            "complete": {
                (3, 1.0): {"grayscale": {"sigma": 5.0, "rho": 0.02}},
                (5, 1.0): {"grayscale": {"sigma": 20.0, "rho": 0.003}},
                (10, 1.0): {"grayscale": {"sigma": 5.0, "rho": 0.01}},
            },
            "partial": {
                "motion": {1.0: {"grayscale": {"sigma": 5.0, "rho": 0.008}}},
                "synthesis": {1.0: {"grayscale": {"sigma": 3.0, "rho": 0.01}}},
            },
        },
    },
    "FedAvgDeblur": {
        "TV": {
            "complete": {
                (10, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.001},
                    "color": {"eta": 0.0001, "rho": 0.001},
                },
                (5, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.0005},
                    "color": {"eta": 0.0001, "rho": 0.0003},
                },
                (3, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.0002},
                    "color": {"eta": 0.0001, "rho": 0.0002},
                },
                (10, 0.5): {
                    "grayscale": {"eta": 0.003, "rho": 0.01},
                    "color": {"eta": 0.005, "rho": 0.01},
                },
                (5, 0.5): {
                    "grayscale": {"eta": 0.005, "rho": 0.01},
                    "color": {"eta": 0.008, "rho": 0.01},
                },
                (3, 0.5): {
                    "grayscale": {"eta": 0.008, "rho": 0.01},
                    "color": {"eta": 0.01, "rho": 0.01},
                },
                (10, 1.0): {
                    "grayscale": {"eta": 0.01, "rho": 0.015},
                    "color": {"eta": 0.015, "rho": 0.015},
                },
                (5, 1.0): {
                    "grayscale": {"eta": 0.02, "rho": 0.015},
                    "color": {"eta": 0.025, "rho": 0.015},
                },
                (3, 1.0): {
                    "grayscale": {"eta": 0.025, "rho": 0.015},
                    "color": {"eta": 0.035, "rho": 0.015},
                },
                (10, 2.0): {
                    "grayscale": {"eta": 0.03, "rho": 0.03},
                    "color": {"eta": 0.05, "rho": 0.03},
                },
                (5, 2.0): {
                    "grayscale": {"eta": 0.075, "rho": 0.05},
                    "color": {"eta": 0.075, "rho": 0.03},
                },
                (3, 2.0): {
                    "grayscale": {"eta": 0.1, "rho": 0.05},
                    "color": {"eta": 0.1, "rho": 0.03},
                },
                (10, 3.0): {
                    "grayscale": {"eta": 0.075, "rho": 0.05},
                    "color": {"eta": 0.1, "rho": 0.05},
                },
                (5, 3.0): {
                    "grayscale": {"eta": 0.2, "rho": 0.3},
                    "color": {"eta": 0.2, "rho": 0.2},
                },
                (3, 3.0): {
                    "grayscale": {"eta": 0.3, "rho": 0.2},
                    "color": {"eta": 0.3, "rho": 0.2},
                },
            },
            "partial": {
                "motion": {
                    0.0: {
                        "grayscale": {"eta": 3.0, "rho": 0.01},
                        "color": {"eta": 3.5, "rho": 0.005},
                    },
                    0.5: {
                        "grayscale": {"eta": 3.0, "rho": 0.01},
                        "color": {"eta": 2.0, "rho": 0.01},
                    },
                    1.0: {
                        "grayscale": {"eta": 3.0, "rho": 0.01},
                        "color": {"eta": 2.5, "rho": 0.005},
                    },
                    2.0: {
                        "grayscale": {"eta": 3.0, "rho": 0.01},
                        "color": {"eta": 2.5, "rho": 0.02},
                    },
                    3.0: {
                        "grayscale": {"eta": 3.0, "rho": 0.01},
                        "color": {"eta": 2.5, "rho": 0.03},
                    },
                },
                "synthesis": {
                    0.0: {
                        "grayscale": {"eta": 3.5, "rho": 0.1},
                        "color": {"eta": 3.0, "rho": 0.1},
                    },
                    0.5: {
                        "grayscale": {"eta": 3.5, "rho": 0.1},
                        "color": {"eta": 3.5, "rho": 0.05},
                    },
                    1.0: {
                        "grayscale": {"eta": 1.5, "rho": 0.2},
                        "color": {"eta": 2.5, "rho": 0.2},
                    },
                    2.0: {
                        "grayscale": {"eta": 3.5, "rho": 0.1},
                        "color": {"eta": 3.0, "rho": 0.1},
                    },
                    3.0: {
                        "grayscale": {"eta": 3.5, "rho": 0.1},
                        "color": {"eta": 2.0, "rho": 0.1},
                    },
                },
            },
        },
        "PnP": {
            "complete": {
                (3, 1.0): {"grayscale": {"sigma": 5.0, "rho": 0.02}},
                (5, 1.0): {"grayscale": {"sigma": 5.0, "rho": 0.02}},
                (10, 1.0): {"grayscale": {"sigma": 1.0, "rho": 0.075}},
            },
            "partial": {
                "motion": {1.0: {"grayscale": {"sigma": 40.0, "rho": 0.2}}},
                "synthesis": {1.0: {"grayscale": {"sigma": 49.0, "rho": 0.2}}},
            },
        },
    },
    "LocDeblur": {
        "TV": {
            "complete": {
                (10, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.0001},
                    "color": {"eta": 0.0001, "rho": 0.0001},
                },
                (5, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.0001},
                    "color": {"eta": 0.0001, "rho": 0.0001},
                },
                (3, 0.0): {
                    "grayscale": {"eta": 0.0001, "rho": 0.0001},
                    "color": {"eta": 0.0001, "rho": 0.0001},
                },
                (10, 0.5): {
                    "grayscale": {"eta": 0.005, "rho": 0.003},
                    "color": {"eta": 0.005, "rho": 0.001},
                },
                (5, 0.5): {
                    "grayscale": {"eta": 0.005, "rho": 0.003},
                    "color": {"eta": 0.008, "rho": 0.003},
                },
                (3, 0.5): {
                    "grayscale": {"eta": 0.005, "rho": 0.005},
                    "color": {"eta": 0.01, "rho": 0.005},
                },
                (10, 1.0): {
                    "grayscale": {"eta": 0.01, "rho": 0.0015},
                    "color": {"eta": 0.02, "rho": 0.0001},
                },
                (5, 1.0): {
                    "grayscale": {"eta": 0.02, "rho": 0.005},
                    "color": {"eta": 0.03, "rho": 0.005},
                },
                (3, 1.0): {
                    "grayscale": {"eta": 0.025, "rho": 0.01},
                    "color": {"eta": 0.05, "rho": 0.01},
                },
                (10, 2.0): {
                    "grayscale": {"eta": 0.05, "rho": 0.0003},
                    "color": {"eta": 0.05, "rho": 0.0002},
                },
                (5, 2.0): {
                    "grayscale": {"eta": 0.08, "rho": 0.01},
                    "color": {"eta": 0.1, "rho": 0.0005},
                },
                (3, 2.0): {
                    "grayscale": {"eta": 0.1, "rho": 0.08},
                    "color": {"eta": 0.1, "rho": 0.02},
                },
                (10, 3.0): {
                    "grayscale": {"eta": 0.1, "rho": 0.0008},
                    "color": {"eta": 0.1, "rho": 0.0005},
                },
                (5, 3.0): {
                    "grayscale": {"eta": 0.1, "rho": 0.03},
                    "color": {"eta": 0.2, "rho": 0.02},
                },
                (3, 3.0): {
                    "grayscale": {"eta": 0.2, "rho": 0.1},
                    "color": {"eta": 0.3, "rho": 0.075},
                },
            },
            "partial": {
                "motion": {
                    0.0: {
                        "grayscale": {"eta": 2.0, "rho": 8.0},
                        "color": {"eta": 3.0, "rho": 5.0},
                    },
                    0.5: {
                        "grayscale": {"eta": 2.0, "rho": 8.0},
                        "color": {"eta": 3.0, "rho": 5.0},
                    },
                    1.0: {
                        "grayscale": {"eta": 2.0, "rho": 8.0},
                        "color": {"eta": 3.0, "rho": 5.0},
                    },
                    2.0: {
                        "grayscale": {"eta": 2.0, "rho": 8.0},
                        "color": {"eta": 2.5, "rho": 5.0},
                    },
                    3.0: {
                        "grayscale": {"eta": 2.0, "rho": 10.0},
                        "color": {"eta": 2.5, "rho": 7.5},
                    },
                },
                "synthesis": {
                    0.0: {
                        "grayscale": {"eta": 2.5, "rho": 8.0},
                        "color": {"eta": 3.0, "rho": 3.5},
                    },
                    0.5: {
                        "grayscale": {"eta": 2.5, "rho": 5.0},
                        "color": {"eta": 2.5, "rho": 3.5},
                    },
                    1.0: {
                        "grayscale": {"eta": 2.5, "rho": 10.0},
                        "color": {"eta": 3.0, "rho": 3.0},
                    },
                    2.0: {
                        "grayscale": {"eta": 2.5, "rho": 2.0},
                        "color": {"eta": 3.0, "rho": 2.5},
                    },
                    3.0: {
                        "grayscale": {"eta": 2.0, "rho": 3.5},
                        "color": {"eta": 3.0, "rho": 3.0},
                    },
                },
            },
        },
        "PnP": {
            "complete": {
                (3, 1.0): {"grayscale": {"sigma": 25.0, "rho": 0.005}},
                (5, 1.0): {"grayscale": {"sigma": 10.0, "rho": 0.01}},
                (10, 1.0): {"grayscale": {"sigma": 25.0, "rho": 0.002}},
            },
            "partial": {
                "motion": {1.0: {"grayscale": {"sigma": 45.0, "rho": 0.3}}},
                "synthesis": {1.0: {"grayscale": {"sigma": 50.0, "rho": 0.5}}},
            },
        },
    },
}


class ParameterManager:
    def __init__(self):
        self.preset_params = PRESET_PARAMS

    def get_params(self, config: dict, is_color: bool = False):
        if config["regularizer"] == "PnP" and is_color:
            raise NotImplementedError("only grayscale images this setup")

        algorithm = config["algorithm"]
        regularizer = config["regularizer"]
        observation = config["observation"]
        noise_std = config["noise_std"]
        image_key = "color" if is_color else "grayscale"

        try:
            alg_params = self.preset_params[algorithm][regularizer][observation]

            if observation == "complete":
                n_clients = config["n_clients"]
                param_key = (n_clients, noise_std)

                if param_key not in alg_params:
                    raise ValueError("No required parameters")

                image_params = alg_params[param_key]

            else:
                blur_type = config["blur_type"]

                if blur_type not in alg_params:
                    raise ValueError("No required parameters")

                blur_params = alg_params[blur_type]
                if noise_std not in blur_params:
                    raise ValueError("No required parameters")

                image_params = blur_params[noise_std]

            if image_key not in image_params:
                raise ValueError("No required parameters")

            final_params = image_params[image_key].copy()
            final_params.setdefault("max_iter", 500)
            final_params.setdefault("tol", 1e-5)
            final_params["gamma"] = 1.05

            if "parameter_overrides" in config:
                overrides = config["parameter_overrides"]

                if "eta" in overrides and regularizer == "TV":
                    final_params["eta"] = overrides["eta"]

                if "rho" in overrides:
                    final_params["rho"] = overrides["rho"]

                if "sigma" in overrides and regularizer == "PnP":
                    final_params["sigma"] = overrides["sigma"]

            return final_params

        except KeyError:
            raise ValueError("not supported")

    def get_available_configs(
        self, algorithm: str = None, regularizer: str = None, observation: str = None
    ) -> list:
        configs = []

        algorithms = [algorithm] if algorithm else self.preset_params.keys()

        for alg in algorithms:
            if alg not in self.preset_params:
                continue

            regularizers = (
                [regularizer] if regularizer else self.preset_params[alg].keys()
            )

            for reg in regularizers:
                if reg not in self.preset_params[alg]:
                    continue

                observations = (
                    [observation]
                    if observation
                    else self.preset_params[alg][reg].keys()
                )

                for obs in observations:
                    if obs not in self.preset_params[alg][reg]:
                        continue

                    if obs == "complete":
                        for n_clients, noise_std in self.preset_params[alg][reg][
                            obs
                        ].keys():
                            configs.append((alg, reg, obs, n_clients, noise_std))
                    else:
                        for blur_type in self.preset_params[alg][reg][obs].keys():
                            for noise_std in self.preset_params[alg][reg][obs][
                                blur_type
                            ].keys():
                                configs.append((alg, reg, obs, blur_type, noise_std))

        return sorted(configs)

    def validate_config(self, config: dict, is_color: bool = False) -> bool:
        try:
            self.get_params(config, is_color)
            return True
        except (ValueError, NotImplementedError):
            return False
