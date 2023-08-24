#include <iostream>
#include <cmath>
#include <autodiff/forward/dual.hpp>
#include <iomanip>
#include <algorithm>
#include <torch/torch.h>

using namespace autodiff;

// Cart-pole parameters
const double g = 9.81;
const double m_c = 1.0;
const double m_p = 0.1;
const double l = 1.0;

// Dynamics function for the cart-pole system with control input
void cartPoleDynamics(dual t, const std::vector<dual> &y, const dual &u, std::vector<dual> &dydt)
{
    dual theta = y[0];
    dual theta_dot = y[1];
    dual x = y[2];
    dual x_dot = y[3];

    dual sin_theta = sin(theta);
    dual cos_theta = cos(theta);

    dual temp1 = (m_p * l * theta_dot * theta_dot * sin_theta) / (m_c + m_p);
    dual temp2 = (g * sin_theta) / (l * (m_c + m_p));

    dydt[0] = theta_dot;
    dydt[1] = (temp2 - temp1 + u * cos_theta / (m_c + m_p)) / (l * (4.0 / 3.0 - m_p * cos_theta * cos_theta / (m_c + m_p)));
    dydt[2] = x_dot;
    dydt[3] = (m_p * l * theta_dot * theta_dot * sin_theta - m_p * g * cos_theta * sin_theta + u) /
              (m_c + m_p);
}

// Fourth-order Runge-Kutta solver
dual rungeKutta(dual t, std::vector<dual> &y, const dual &u, dual h, const std::array<dual, 4> &y_hat)
{
    std::vector<dual> k1(4), k2(4), k3(4), k4(4);
    std::vector<dual> y_temp(4);

    cartPoleDynamics(t, y, u, k1);
    for (int i = 0; i < 4; ++i)
        y_temp[i] = y[i] + h * 0.5 * k1[i];

    cartPoleDynamics(t + h * 0.5, y_temp, u, k2);
    for (int i = 0; i < 4; ++i)
        y_temp[i] = y[i] + h * 0.5 * k2[i];

    cartPoleDynamics(t + h * 0.5, y_temp, u, k3);
    for (int i = 0; i < 4; ++i)
        y_temp[i] = y[i] + h * k3[i];

    cartPoleDynamics(t + h, y_temp, u, k4);
    for (int i = 0; i < 4; ++i)
        y[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);

    // L2 norm of error y-y_hat

    dual loss = 0.0;
    // for (int i = 0; i < 4; ++i)
    // {
    loss += (y[0] - y_hat[0]) * (y[0] - y_hat[0]);
    loss += (y[2] - y_hat[2]) * (y[2] - y_hat[2]);
    // }

    return loss;
}

// Define a new Module.
struct Net : torch::nn::Module
{
    Net()
    {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(4, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 16));
        fc4 = register_module("fc4", torch::nn::Linear(16, 1));
        // init weights
        torch::nn::init::uniform_(fc1->weight, -0.1, 0.1);
        torch::nn::init::uniform_(fc2->weight, -0.1, 0.1);
        torch::nn::init::uniform_(fc3->weight, -0.1, 0.1);
        torch::nn::init::uniform_(fc4->weight, -0.1, 0.1);
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        x = torch::gelu(fc1->forward(x.reshape({x.size(0), 4})));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::gelu(fc2->forward(x));
        x = torch::gelu(fc3->forward(x));
        x = torch::tanh(fc4->forward(x));
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr},
        fc3{nullptr}, fc4{nullptr};
};

int main()
{
    at::set_num_threads(1);
    std::cout << "Threads: " << at::get_num_threads() << std::endl;

    constexpr double USCALE = 10.0; // umax
    auto net = std::make_shared<Net>();
    torch::optim::AdamW optimizer(net->parameters(), /*lr=*/0.0001);

#if 0
    for (int j = 0; j < 10000; j++)
    {
        // reset simulation
        dual t = 0.0;
        dual h = 0.001;
        std::vector<dual> y = {0.0, 0.0, 0.0, 0.0}; // initial conditions
        y[0] = 0.1 * (rand() % 5 - 2);
        dual u = 0.0;
        std::array<dual, 4> y_hat = {0.0, 0.0, 0.0, 0.0}; // target state
        int i;
        double avg_loss = 0.0;
        for (i = 0; i < 10000; ++i)
        {
            optimizer.zero_grad();

            torch::Tensor tensor = torch::zeros({1, 4});
            for (int k = 0; k < 4; ++k)
                tensor[0][k] = y[k].val;
            torch::Tensor prediction = net->forward(tensor);
            u = prediction.item<double>() * USCALE;

            dual loss = rungeKutta(t, y, u, h, y_hat);
            avg_loss += loss.val;
            t += h;

            if (std::abs(y[0].val) > 0.5)
                break;

            double dydu = derivative(rungeKutta, wrt(u), autodiff::at(t, y, u, h, y_hat));

            if (std::isnan(dydu) || std::isnan(loss.val) || std::isnan(u.val))
            {
                exit(1);
            }

            torch::Tensor gradient = torch::zeros({1, 1});
            gradient[0][0] = dydu;
            prediction.backward(gradient);
            optimizer.step();
        }
        avg_loss /= i;
        std::cout << "Steps: " << i << " Loss: " << avg_loss << std::endl;
        if (i > 9000 && avg_loss < 0.05)
            break;
    }
    // save weights
    torch::save(net, "net.pt");
#else
    torch::load(net, "net.pt");
#endif

    // test out and print states along trajectory
    dual t = 0.0;
    dual h = 0.001;
    std::vector<dual> y = {0.0, 0.0, 0.0, 0.0}; // initial conditions
    dual u = 0.0;
    std::array<dual, 4> y_hat = {0.0, 0.0, 0.0, 0.0}; // target state
    for (int i = 0; i < 20001; ++i)
    {
        torch::Tensor tensor = torch::zeros({1, 4});
        for (int k = 0; k < 4; ++k)
            tensor[0][k] = y[k].val;
        torch::Tensor prediction = net->forward(tensor);
        u = prediction.item<double>() * USCALE;

        rungeKutta(t, y, u, h, y_hat);
        t += h;

        std::cout << std::fixed << std::setprecision(5) << "theta:\t" << y[0].val
                  << " theta_dot:\t" << y[1].val << " x:\t" << y[2].val
                  << " x_dot:\t" << y[3].val << " input: " << u << std::endl;
    }

    return 0;
}
