defmodule NxImage.MixProject do
  use Mix.Project

  @version "0.1.2"
  @description "Image processing in Nx"

  def project do
    [
      app: :nx_image,
      version: @version,
      description: @description,
      name: "NxImage",
      elixir: "~> 1.13",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      package: package()
    ]
  end

  def application do
    []
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, "~> 0.4"},
      {:ex_doc, "~> 0.29", only: :dev, runtime: false}
    ]
  end

  defp docs do
    [
      main: "NxImage",
      source_url: "https://github.com/elixir-nx/nx_image",
      source_ref: "v#{@version}",
      extras: ["notebooks/examples.livemd"],
      groups_for_functions: [
        Transformation: &(&1[:type] == :transformation),
        Conversion: &(&1[:type] == :conversion)
      ]
    ]
  end

  def package do
    [
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => "https://github.com/elixir-nx/nx_image"
      }
    ]
  end
end
