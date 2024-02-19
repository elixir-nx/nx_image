defmodule NxImageTest do
  use ExUnit.Case

  import NxImage.TestHelpers

  doctest NxImage

  describe "resize/3" do
    test "methods" do
      # Reference values computed in jax

      image = Nx.iota({2, 2, 3}, type: :f32)

      assert NxImage.resize(image, {3, 3}, method: :nearest) ==
               Nx.tensor([
                 [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
                 [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
                 [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0], [9.0, 10.0, 11.0]]
               ])

      assert NxImage.resize(image, {3, 3}, method: :bilinear) ==
               Nx.tensor([
                 [[0.0, 1.0, 2.0], [1.5, 2.5, 3.5], [3.0, 4.0, 5.0]],
                 [[3.0, 4.0, 5.0], [4.5, 5.5, 6.5], [6.0, 7.0, 8.0]],
                 [[6.0, 7.0, 8.0], [7.5, 8.5, 9.5], [9.0, 10.0, 11.0]]
               ])

      assert_all_close(
        NxImage.resize(image, {3, 3}, method: :bicubic),
        Nx.tensor([
          [[-0.5921, 0.4079, 1.4079], [1.1053, 2.1053, 3.1053], [2.8026, 3.8026, 4.8026]],
          [[2.8026, 3.8026, 4.8026], [4.5, 5.5, 6.5], [6.1974, 7.1974, 8.1974]],
          [[6.1974, 7.1974, 8.1974], [7.8947, 8.8947, 9.8947], [9.5921, 10.5921, 11.5921]]
        ])
      )

      assert_all_close(
        NxImage.resize(image, {3, 3}, method: :lanczos3),
        Nx.tensor([
          [[-1.1173, -0.1173, 0.8827], [0.7551, 1.7551, 2.7551], [2.6276, 3.6276, 4.6276]],
          [[2.6276, 3.6276, 4.6276], [4.5, 5.5, 6.5], [6.3724, 7.3724, 8.3724]],
          [[6.3724, 7.3724, 8.3724], [8.2449, 9.2449, 10.2449], [10.1173, 11.1173, 12.1173]]
        ])
      )

      assert_all_close(
        NxImage.resize(image, {3, 3}, method: :lanczos5),
        Nx.tensor([
          [[-1.3525, -0.3525, 0.6475], [0.5984, 1.5984, 2.5984], [2.5492, 3.5492, 4.5492]],
          [[2.5492, 3.5492, 4.5492], [4.5, 5.5, 6.5], [6.4508, 7.4508, 8.4508]],
          [[6.4508, 7.4508, 8.4508], [8.4016, 9.4016, 10.4016], [10.3525, 11.3525, 12.3525]]
        ])
      )
    end

    test "without anti-aliasing" do
      # Upscaling

      image = Nx.iota({4, 4, 3}, type: :f32)

      assert_all_close(
        NxImage.resize(image, {3, 3}, method: :bicubic, antialias: false),
        Nx.tensor([
          [
            [[1.5427, 2.5427, 3.5427], [5.7341, 6.7341, 7.7341], [9.9256, 10.9256, 11.9256]],
            [[18.3085, 19.3085, 20.3085], [22.5, 23.5, 24.5], [26.6915, 27.6915, 28.6915]],
            [
              [35.0744, 36.0744, 37.0744],
              [39.2659, 40.2659, 41.2659],
              [43.4573, 44.4573, 45.4573]
            ]
          ]
        ])
      )

      # Downscaling (no effect)

      image = Nx.iota({2, 2, 3}, type: :f32)

      assert_all_close(
        NxImage.resize(image, {3, 3}, method: :bicubic, antialias: false),
        Nx.tensor([
          [[-0.5921, 0.4079, 1.4079], [1.1053, 2.1053, 3.1053], [2.8026, 3.8026, 4.8026]],
          [[2.8026, 3.8026, 4.8026], [4.5, 5.5, 6.5], [6.1974, 7.1974, 8.1974]],
          [[6.1974, 7.1974, 8.1974], [7.8947, 8.8947, 9.8947], [9.5921, 10.5921, 11.5921]]
        ])
      )
    end

    test "accepts a batch" do
      image = Nx.iota({2, 2, 3}, type: :f32)
      resized_image = NxImage.resize(image, {3, 3})

      batch_once = fn x -> Nx.stack([x, x]) end
      assert NxImage.resize(batch_once.(image), {3, 3}) == batch_once.(resized_image)

      batch_twice = fn x -> batch_once.(batch_once.(x)) end
      assert NxImage.resize(batch_twice.(image), {3, 3}) == batch_twice.(resized_image)
    end

    test "supports with channels-first" do
      image = Nx.iota({2, 2, 3}, type: :f32)
      resized_image = NxImage.resize(image, {3, 3})

      to_channels_first = fn x -> Nx.transpose(x, axes: [2, 0, 1]) end

      assert NxImage.resize(to_channels_first.(image), {3, 3}, channels: :first) ==
               to_channels_first.(resized_image)
    end
  end
end
