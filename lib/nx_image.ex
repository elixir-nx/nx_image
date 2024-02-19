defmodule NxImage do
  @moduledoc """
  Image processing in `Nx`.

  All functions expect images to be tensors in either HWC or CHW order,
  with an arbitrary number of leading batch axes.

  All transformations preserve the input type, rounding if necessary.
  For higher precision, cast the input to floating-point beforehand.
  """

  import Nx.Defn

  @doc """
  Crops an image at the center.

  If the image is too small to be cropped to the desired size, it gets
  padded with zeros.

  ## Options

    * `:channels` - channels location, either `:first` or `:last`.
      Defaults to `:last`

  ## Examples

      iex> image = Nx.iota({4, 4, 1}, type: :u8)
      iex> NxImage.center_crop(image, {2, 2})
      #Nx.Tensor<
        u8[2][2][1]
        [
          [
            [5],
            [6]
          ],
          [
            [9],
            [10]
          ]
        ]
      >

      iex> image = Nx.iota({2, 2, 1}, type: :u8)
      iex> NxImage.center_crop(image, {1, 4})
      #Nx.Tensor<
        u8[1][4][1]
        [
          [
            [0],
            [0],
            [1],
            [0]
          ]
        ]
      >

  """
  @doc type: :transformation
  deftransform center_crop(input, size, opts \\ []) when is_tuple(size) do
    opts = Keyword.validate!(opts, channels: :last)
    validate_image!(input)

    pad_config =
      for {axis, size, out_size} <- spatial_axes_with_sizes(input, size, opts[:channels]),
          reduce: List.duplicate({0, 0, 0}, Nx.rank(input)) do
        pad_config ->
          low = div(size - out_size, 2)
          high = low + out_size
          List.replace_at(pad_config, axis, {-low, high - size, 0})
      end

    Nx.pad(input, 0, pad_config)
  end

  deftransformp spatial_axes_with_sizes(input, size, channels) do
    {height_axis, width_axis} = spatial_axes(input, channels)
    {height, width} = size(input, channels)
    {out_height, out_width} = size
    [{height_axis, height, out_height}, {width_axis, width, out_width}]
  end

  # Returns the image size as `{height, width}`.
  deftransformp size(input, channels) do
    {height_axis, width_axis} = spatial_axes(input, channels)
    {Nx.axis_size(input, height_axis), Nx.axis_size(input, width_axis)}
  end

  @doc """
  Resizes an image.

  ## Options

    * `:method` - the resizing method to use, either of `:nearest`,
      `:bilinear`, `:bicubic`, `:lanczos3`, `:lanczos5`. Defaults to
      `:bilinear`

    * `:antialias` - whether an anti-aliasing filter should be used
      when downsampling. This has no effect with upsampling. Defaults
      to `true`

    * `:channels` - channels location, either `:first` or `:last`.
      Defaults to `:last`

  ## Examples

      iex> image = Nx.iota({2, 2, 1}, type: :u8)
      iex> NxImage.resize(image, {3, 3}, method: :nearest)
      #Nx.Tensor<
        u8[3][3][1]
        [
          [
            [0],
            [1],
            [1]
          ],
          [
            [2],
            [3],
            [3]
          ],
          [
            [2],
            [3],
            [3]
          ]
        ]
      >

      iex> image = Nx.iota({2, 2, 1}, type: :f32)
      iex> NxImage.resize(image, {3, 3}, method: :bilinear)
      #Nx.Tensor<
        f32[3][3][1]
        [
          [
            [0.0],
            [0.5],
            [1.0]
          ],
          [
            [1.0],
            [1.5],
            [2.0]
          ],
          [
            [2.0],
            [2.5],
            [3.0]
          ]
        ]
      >

  """
  @doc type: :transformation
  deftransform resize(input, size, opts \\ []) when is_tuple(size) do
    opts = Keyword.validate!(opts, channels: :last, method: :bilinear, antialias: true)
    validate_image!(input)

    {spatial_axes, out_shape} =
      input
      |> spatial_axes_with_sizes(size, opts[:channels])
      |> Enum.reject(fn {_axis, size, out_size} -> Elixir.Kernel.==(size, out_size) end)
      |> Enum.map_reduce(Nx.shape(input), fn {axis, _size, out_size}, out_shape ->
        {axis, put_elem(out_shape, axis, out_size)}
      end)

    antialias = opts[:antialias]

    resized_input =
      case opts[:method] do
        :nearest ->
          resize_nearest(input, out_shape, spatial_axes)

        :bilinear ->
          resize_with_kernel(input, out_shape, spatial_axes, antialias, &fill_linear_kernel/1)

        :bicubic ->
          resize_with_kernel(input, out_shape, spatial_axes, antialias, &fill_cubic_kernel/1)

        :lanczos3 ->
          resize_with_kernel(
            input,
            out_shape,
            spatial_axes,
            antialias,
            &fill_lanczos_kernel(3, &1)
          )

        :lanczos5 ->
          resize_with_kernel(
            input,
            out_shape,
            spatial_axes,
            antialias,
            &fill_lanczos_kernel(5, &1)
          )

        method ->
          raise ArgumentError,
                "expected :method to be either of :nearest, :bilinear, :bicubic, " <>
                  ":lanczos3, :lanczos5, got: #{inspect(method)}"
      end

    cast_to(resized_input, input)
  end

  deftransformp spatial_axes(input, channels) do
    axes =
      case channels do
        :first -> [-2, -1]
        :last -> [-3, -2]
      end

    axes
    |> Enum.map(&Nx.axis_index(input, &1))
    |> List.to_tuple()
  end

  defnp cast_to(left, right) do
    left_type = Nx.type(left)
    right_type = Nx.type(right)

    left =
      if Nx.Type.float?(left_type) and Nx.Type.integer?(right_type) do
        Nx.round(left)
      else
        left
      end

    left
    |> Nx.as_type(right_type)
    |> Nx.reshape(left, names: Nx.names(right))
  end

  deftransformp resize_nearest(input, out_shape, spatial_axes) do
    singular_shape = List.duplicate(1, Nx.rank(input)) |> List.to_tuple()

    for axis <- spatial_axes, reduce: input do
      input ->
        input_shape = Nx.shape(input)
        input_size = elem(input_shape, axis)
        output_size = elem(out_shape, axis)
        inv_scale = input_size / output_size
        offset = Nx.iota({output_size}) |> Nx.add(0.5) |> Nx.multiply(inv_scale)
        offset = offset |> Nx.floor() |> Nx.as_type({:s, 32})

        offset =
          offset
          |> Nx.reshape(put_elem(singular_shape, axis, output_size))
          |> Nx.broadcast(put_elem(input_shape, axis, output_size))

        Nx.take_along_axis(input, offset, axis: axis)
    end
  end

  @f32_eps :math.pow(2, -23)

  deftransformp resize_with_kernel(input, out_shape, spatial_axes, antialias, kernel_fun) do
    for axis <- spatial_axes, reduce: input do
      input ->
        resize_axis_with_kernel(input,
          axis: axis,
          output_size: elem(out_shape, axis),
          antialias: antialias,
          kernel_fun: kernel_fun
        )
    end
  end

  defnp resize_axis_with_kernel(input, opts) do
    axis = opts[:axis]
    output_size = opts[:output_size]
    antialias = opts[:antialias]
    kernel_fun = opts[:kernel_fun]

    input_size = Nx.axis_size(input, axis)

    inv_scale = input_size / output_size

    kernel_scale =
      if antialias do
        max(1, inv_scale)
      else
        1
      end

    sample_f = (Nx.iota({1, output_size}) + 0.5) * inv_scale - 0.5
    x = Nx.abs(sample_f - Nx.iota({input_size, 1})) / kernel_scale
    weights = kernel_fun.(x)

    weights_sum = Nx.sum(weights, axes: [0], keep_axes: true)

    weights = Nx.select(Nx.abs(weights) > 1000 * @f32_eps, safe_divide(weights, weights_sum), 0)

    input = Nx.dot(input, [axis], weights, [0])
    # The transformed axis is moved to the end, so we transpose back
    reorder_axis(input, -1, axis)
  end

  defnp fill_linear_kernel(x) do
    Nx.max(0, 1 - x)
  end

  defnp fill_cubic_kernel(x) do
    # See https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    out = (1.5 * x - 2.5) * x * x + 1
    out = Nx.select(x >= 1, ((-0.5 * x + 2.5) * x - 4) * x + 2, out)
    Nx.select(x >= 2, 0, out)
  end

  @pi :math.pi()

  defnp fill_lanczos_kernel(radius, x) do
    y = radius * Nx.sin(@pi * x) * Nx.sin(@pi * x / radius)
    out = Nx.select(x > 1.0e-3, safe_divide(y, @pi ** 2 * x ** 2), 1)
    Nx.select(x > radius, 0, out)
  end

  defnp safe_divide(x, y) do
    x / Nx.select(y != 0, y, 1)
  end

  deftransformp reorder_axis(tensor, axis, target_axis) do
    axes = Nx.axes(tensor)
    {source_axis, axes} = List.pop_at(axes, axis)
    axes = List.insert_at(axes, target_axis, source_axis)
    Nx.transpose(tensor, axes: axes)
  end

  @doc """
  Scales an image such that the short edge matches the given size.

  ## Options

    * `:method` - the resizing method to use, same as `resize/2`

    * `:antialias` - whether an anti-aliasing filter should be used
      when downsampling. This has no effect with upsampling. Defaults
      to `true`

    * `:channels` - channels location, either `:first` or `:last`.
      Defaults to `:last`

  ## Examples

      iex> image = Nx.iota({2, 4, 1}, type: :u8)
      iex> resized_image = NxImage.resize_short(image, 3, method: :nearest)
      iex> Nx.shape(resized_image)
      {3, 6, 1}

      iex> image = Nx.iota({4, 2, 1}, type: :u8)
      iex> resized_image = NxImage.resize_short(image, 3, method: :nearest)
      iex> Nx.shape(resized_image)
      {6, 3, 1}

  """
  @doc type: :transformation
  deftransform resize_short(input, size, opts \\ []) when is_integer(size) do
    opts = Keyword.validate!(opts, channels: :last, method: :bilinear, antialias: true)
    validate_image!(input)
    resize_short_n(input, [size: size] ++ opts)
  end

  defnp resize_short_n(input, opts) do
    size = opts[:size]
    method = opts[:method]
    antialias = opts[:antialias]
    channels = opts[:channels]

    {height, width} = size(input, channels)
    {out_height, out_width} = resize_short_size(height, width, size)

    resize(input, {out_height, out_width},
      method: method,
      antialias: antialias,
      channels: channels
    )
  end

  deftransformp resize_short_size(height, width, size) do
    {short, long} = if height < width, do: {height, width}, else: {width, height}

    out_short = size
    out_long = floor(size * long / short)

    if height < width, do: {out_short, out_long}, else: {out_long, out_short}
  end

  @doc """
  Normalizes an image according to the given per-channel mean and
  standard deviation.

    * `:channels` - channels location, either `:first` or `:last`.
      Defaults to `:last`

  ## Examples

      iex> image = Nx.iota({2, 2, 3}, type: :f32)
      iex> mean = Nx.tensor([0.485, 0.456, 0.406])
      iex> std = Nx.tensor([0.229, 0.224, 0.225])
      iex> NxImage.normalize(image, mean, std)
      #Nx.Tensor<
        f32[2][2][3]
        [
          [
            [-2.1179039478302, 2.4285714626312256, 7.084444522857666],
            [10.982532501220703, 15.821427345275879, 20.41777801513672]
          ],
          [
            [24.08296775817871, 29.214284896850586, 33.7511100769043],
            [37.183406829833984, 42.607139587402344, 47.08444595336914]
          ]
        ]
      >

  """
  @doc type: :transformation
  defn normalize(input, mean, std, opts \\ []) do
    opts = keyword!(opts, channels: :last)
    validate_image!(input)

    mean = broadcast_channel_info(mean, input, opts[:channels], "mean")
    std = broadcast_channel_info(std, input, opts[:channels], "std")

    normalized_input = (input - mean) / std

    cast_to(normalized_input, input)
  end

  deftransformp broadcast_channel_info(tensor, input, channels, name) do
    rank = Nx.rank(input)

    channels_axis =
      case channels do
        :first -> rank - 3
        :last -> rank - 1
      end

    num_channels = Nx.axis_size(input, channels_axis)

    case Nx.shape(tensor) do
      {^num_channels} ->
        :ok

      shape ->
        raise ArgumentError,
              "expected #{name} to have shape {#{num_channels}}, got: #{inspect(shape)}"
    end

    shape = 1 |> Tuple.duplicate(rank) |> put_elem(channels_axis, :auto)
    Nx.reshape(tensor, shape)
  end

  @doc """
  Converts pixel values (0-255) into a continuous range.

  ## Examples

      iex> image = Nx.tensor([[[0], [128]], [[191], [255]]])
      iex> NxImage.to_continuous(image, 0.0, 1.0)
      #Nx.Tensor<
        f32[2][2][1]
        [
          [
            [0.0],
            [0.501960813999176]
          ],
          [
            [0.7490196228027344],
            [1.0]
          ]
        ]
      >

      iex> image = Nx.tensor([[[0], [128]], [[191], [255]]])
      iex> NxImage.to_continuous(image, -1.0, 1.0)
      #Nx.Tensor<
        f32[2][2][1]
        [
          [
            [-1.0],
            [0.003921627998352051]
          ],
          [
            [0.49803924560546875],
            [1.0]
          ]
        ]
      >

  """
  @doc type: :conversion
  defn to_continuous(input, min, max) do
    validate_image!(input)

    input / 255.0 * (max - min) + min
  end

  @doc """
  Converts values from continuous range into pixel values (0-255).

  ## Examples

      iex> image = Nx.tensor([[[0.0], [0.5]], [[0.75], [1.0]]])
      iex> NxImage.from_continuous(image, 0.0, 1.0)
      #Nx.Tensor<
        u8[2][2][1]
        [
          [
            [0],
            [128]
          ],
          [
            [191],
            [255]
          ]
        ]
      >

      iex> image = Nx.tensor([[[-1.0], [0.0]], [[0.5], [1.0]]])
      iex> NxImage.from_continuous(image, -1.0, 1.0)
      #Nx.Tensor<
        u8[2][2][1]
        [
          [
            [0],
            [128]
          ],
          [
            [191],
            [255]
          ]
        ]
      >

  """
  @doc type: :conversion
  defn from_continuous(input, min, max) do
    validate_image!(input)

    input = (input - min) / (max - min) * 255.0

    input
    |> Nx.round()
    |> Nx.clip(0, 255)
    |> Nx.as_type(:u8)
  end

  deftransformp validate_image!(input) do
    rank = Nx.rank(input)

    if rank < 3 do
      raise ArgumentError,
            "expected the image input to have rank 3 or higher, got: #{inspect(rank)}"
    end
  end
end
