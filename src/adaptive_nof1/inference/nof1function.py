from adaptive_nof1.helpers import series_to_indexed_array
import numpy
import pymc

try:
    from rpy2.robjects.packages import importr
    import rpy2.robjects.packages as rpackages
    import rpy2.robjects as robjects

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    has_rpy2 = True
except ImportError:
    has_rpy2 = False


if has_rpy2:
    # Class implementing a callout to R sampling code from the library nof1
    # Source code can be found at https://github.com/joannamwalsh/nofone
    # Installation requires installing r2py.
    class Nof1Function:
        def __init__(
            self, treatment_name="treatment", outcome_name="outcome", seed=None
        ):
            self.probabilities = [0.5, 0.5]
            self.setup_r_function()
            self.treatment_name = treatment_name
            self.outcome_name = outcome_name

        def setup_r_function(self):
            utils = rpackages.importr("utils")
            utils.chooseCRANmirror(ind=1)
            utils.install_packages("~/Documents/code/nofone", type="source")
            rpackages.importr("nofone")
            robjects.r(
                """
                a <- seq(1, 2)
                # paindata <- read.csv("one_patient.csv")
                # paindata$treatment <- as.character(paindata$treatment)
                model_function <- function(df) {
                    onepainexample <- nof1function(data = df,
                                           ID              = "patient_id",
                                           groupvars       = c(),
                                           stratvars       = c(),
                                           Y               = "outcome",
                                           treatment       = "treatment",
                                           corr.y          = F,
                                           bs.trend        = F,
                                           y.time          = NULL,
                                           knots.bt.block  = NULL,
                                           block.no        = NULL,
                                           bs.df           = NULL,
                                           beta.prior      = list("dunif", 10, 20),
                                           hy.prior        = list("dunif", 0, 1000),
                                           n.chains        = 3,
                                           max.run         = 10^8,
                                           setsize         = 10^3,
                                           n.run           = 100,
                                           conv.limit      = 1.1,
                                           clinicaldiff    = 0.0,
                                           a              = a)
                    onepainexample[["indivsummary"]]
                }
            """
            )
            self.model_function = robjects.r["model_function"]

        def update_posterior(self, history, number_of_treatments):
            assert (
                number_of_treatments == 2
            ), "Current implementation only supports 2 treatments"
            self.history = history

            df = history.to_df()[[self.outcome_name, "patient_id", self.treatment_name]]
            if len(df) < 10:
                return

            # currently nof1function requires str for treatment variable
            df["treatment"] = df["treatment"].astype(str)

            with (ro.default_converter + pandas2ri.converter).context():
                result = self.model_function(df)
            self.probabilities = result[["firstbetter", "secondbetter"]].values[0]
            print(self.probabilities)

        def debug_data(self):
            return {"probabilities": self.probabilities}

        def __str__(self):
            return "Nof1Function"

        def approximate_max_probabilities(self, number_of_treatments, context):
            assert (
                number_of_treatments == 2
            ), "Current implementation only supports 2 treatments"
            return self.probabilities
