import param
from functools import reduce
from scipy.stats import norm
import numpy
import holoviews
import panel
import pandas
from adaptive_nof1.helpers import flatten_dictionary
import copy


def show_patient_explorer(calculated_series, filter_attributes=None):
    global calculated_series_for_explorer
    calculated_series_for_explorer = copy.deepcopy(calculated_series)

    # Inline definition, as we need the value of calculated_series already when defining the class
    class PatientExplorer(param.Parameterized):
        calculated_series = calculated_series_for_explorer
        df = calculated_series[0]["result"].simulations[0].history.to_df()
        patient_id = param.Integer(default=1, bounds=(1, 20))
        configuration = param.Integer(default=0, bounds=(0, len(calculated_series) - 1))
        t = param.Integer(default=0, bounds=df["t"].agg(["min", "max"]))

        @param.depends("patient_id", "configuration")
        def hvplot(self):
            debug_data = (
                self.calculated_series[self.configuration]["result"]
                .simulations[self.patient_id - 1]
                .history.debug_data()
            )
            df = pandas.DataFrame([flatten_dictionary(d) for d in debug_data])
            if filter_attributes is not None:
                df = df.drop(columns=filter_attributes, errors="ignore")
            return df.hvplot()

        @param.depends("configuration")
        def configuration_name(self):
            return panel.panel(
                self.calculated_series[self.configuration]["configuration"]
            )

        @param.depends("patient_id", "configuration", "t")
        def posterior(self):
            debug_data = (
                self.calculated_series[self.configuration]["result"]
                .simulations[self.patient_id - 1]
                .history.debug_data()
            )
            posterior_parameters = debug_data[self.t].copy()
            if "probabilities" in posterior_parameters:
                del posterior_parameters["probabilities"]
            posterior = None
            additional_config = (
                self.calculated_series[self.configuration]["result"]
                .simulations[0]
                .additional_config
            )
            if (
                "inference" in additional_config
                and additional_config["inference"][0:19] == "NormalKnownVariance"
            ):
                posterior = lambda x: norm(
                    posterior_parameters["mean"][x], posterior_parameters["variance"][x]
                )
            if posterior:
                curves = []
                for intervention in range(self.df["treatment"].max() + 1):
                    rv = posterior(intervention)
                    x = numpy.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
                    curves += [zip(x, rv.pdf(x))]
                return reduce(
                    lambda x, y: x * y, [holoviews.Curve(curve) for curve in curves]
                ).opts(xlim=(-3, 3))

    explorer = PatientExplorer()
    hvplot = holoviews.DynamicMap(explorer.hvplot)

    def adjust_non_selected_glyph(plot, element):
        # Adjusting the non-selected glyphs to be fully transparent
        for renderer in plot.state.renderers:
            renderer.nonselection_glyph.line_alpha = 0.0
            if hasattr(renderer, "glyph"):
                # Setting non-selected glyph properties
                renderer.nonselection_glyph.line_alpha = 0.0

    # Applying the hook to the plot
    hvplot = hvplot.opts(hooks=[adjust_non_selected_glyph])

    return panel.Column(
        panel.Row(panel.Column(explorer.param, explorer.configuration_name), hvplot),
        explorer.posterior,
    )
