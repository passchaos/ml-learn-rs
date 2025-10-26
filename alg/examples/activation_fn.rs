use plotiron::prelude::*;
use vectra::prelude::*;

fn main() {
    let input = Array::arange(-5.0, 5.0, 0.01);

    let y_sigmoid = input.sigmoid();
    let y_relu = input.relu();

    let mut fig = figure();
    let axes1 = fig.add_subplot();

    axes1.add_plot(Plot::scatter(input.clone(), y_sigmoid).label("sigmoid"));

    let axes2 = fig.add_subplot();
    axes2.add_plot(Plot::scatter(input, y_relu).label("relu"));

    fig.show();
}
