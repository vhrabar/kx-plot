"""Image input (kx/_image.py) and the image kinds."""
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

import kx
import kx._image
from kx._image import images, show, tile

RGB = np.random.default_rng(0).integers(0, 256, (5, 8, 6, 3), dtype=np.uint8)


# ---------- image input ----------
@pytest.mark.parametrize(("x", "shapes"), [
    (RGB, [(8, 6, 3)] * 5),                                   # (N, H, W, C)
    (RGB[..., 0], [(8, 6)] * 5),                               # (N, H, W): grey
    (RGB[0, ..., 0], [(8, 6)]),                                # one 2-D image
    (RGB.transpose(0, 3, 1, 2), [(8, 6, 3)] * 5),              # (N, C, H, W), as PyTorch has it
    (list(RGB[:2]), [(8, 6, 3)] * 2),                          # a list of arrays
    ([RGB[0], RGB[1, :4, :3]], [(8, 6, 3), (4, 3, 3)]),        # sizes may differ
    ([RGB[0, ..., :1]], [(8, 6)]),                             # (H, W, 1) is grey
    ([np.dstack([RGB[0], np.full((8, 6), 255, np.uint8)])], [(8, 6, 3)]),   # alpha dropped
])
def test_images_shapes(x: object, shapes: list[tuple[int, ...]]) -> None:
    out = images(x)
    assert [a.shape for a in out] == shapes
    assert all(a.dtype == float and a.min() >= 0 and a.max() <= 1 for a in out)


def test_images_channels_first_keeps_pixels() -> None:
    assert np.array_equal(images(RGB.transpose(0, 3, 1, 2))[2], RGB[2] / 255)


@pytest.mark.parametrize(("image", "want"), [
    (np.array([[0, 255]], dtype=np.uint8), [0, 1]),
    (np.array([[0, 65535]], dtype=np.uint16), [0, 1]),
    (np.array([[0, 255]]), [0, 1]),                          # plain ints up to 255 are 8-bit
    (np.array([[0, 1000]]), [0, 1000 / 65535]),              # above 255: 16-bit
    (np.array([[0.2, 0.6]]), [0.2, 0.6]),                    # floats in 0..1 are kept
    (np.array([[-1.0, 0.0, 3.0]]), [0, 0.25, 1]),            # outside 0..1: scaled to fill it
    (np.array([[2.0, 2.0]]), [0, 0]),                        # constant outside 0..1: black, no division by 0
    (np.array([[True, False]]), [1, 0]),
])
def test_images_scale_to_0_1(image: np.ndarray, want: list[float]) -> None:
    assert images(image)[0].ravel() == pytest.approx(want)


def test_images_reads_paths(tmp_path: Path) -> None:
    paths = []
    for i in range(2):
        paths.append(tmp_path / f"{i}.png")
        plt.imsave(paths[-1], RGB[i])
    out = images([*paths[:1], str(paths[1])])                # Path or str
    assert [a.shape for a in out] == [(8, 6, 3)] * 2         # PNG alpha dropped
    assert np.allclose(out[1], RGB[1] / 255, atol=1 / 255)
    assert len(images(str(paths[0]))) == 1                   # one path is one image


def test_images_takes_the_shapes_dataset() -> None:
    out = images(kx.load("shapes")["images"])
    assert len(out) == 210
    assert out[0].shape == (40, 40, 3)


@pytest.mark.parametrize(("x", "match"), [
    ([], "no images"),
    (np.zeros((2, 3, 4, 5, 3)), "one array"),
    (np.zeros(5), "one array"),
    ([np.zeros((4, 4, 2))], "image 0 .* 1, 3 or 4 channels"),
    ([np.zeros((4, 4)), np.zeros(4)], "image 1 has shape"),
])
def test_images_errors(x: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        images(x)


def test_tile_lays_out_n_bare_panels_in_ax_place() -> None:
    left, right = kx.grid(2)
    axs = tile(right, 5, cols=3)
    fig = left.figure
    assert len(axs) == 5
    assert right not in fig.axes
    assert len(fig.axes) == 1 + 5                            # the 6th cell of the 2 x 3 block is removed
    a = axs[0]
    assert (list(a.get_xticks()), list(a.get_yticks())) == ([], [])
    assert not any(s.get_visible() for s in a.spines.values())
    a.set_ylabel("blob")
    assert a.yaxis.label.get_visible()                       # room for a class name at the left


def test_tile_is_about_square_by_default() -> None:
    axs = tile(plt.subplots()[1], 7)
    gs = axs[0].get_subplotspec().get_gridspec()
    assert (gs.nrows, gs.ncols) == (3, 3)
    with pytest.raises(ValueError, match="no images"):
        tile(plt.subplots()[1], 0)


def test_show_grey_and_colour_on_the_full_scale() -> None:
    ax = plt.subplots()[1]
    grey = show(ax, images(RGB[0, ..., 0])[0])
    colour = show(ax, images(RGB[:1])[0], interpolation="bilinear")
    assert grey.get_cmap().name == "gray"
    assert grey.get_clim() == (0, 1)
    assert colour.get_interpolation() == "bilinear"          # keywords win


# ---------- imgrid ----------
SHAPES = kx.load("shapes")


def _pixels(ax: plt.Axes) -> np.ndarray:
    return np.asarray(ax.images[0].get_array())


def test_imgrid_one_row_of_n_per_class_named_at_the_left() -> None:
    axs = kx.plot(SHAPES["images"], SHAPES["labels"], "imgrid", n=4)
    classes = sorted(set(SHAPES["labels"]))
    assert len(axs) == 4 * len(classes)
    firsts = axs[::4]
    assert [a.get_ylabel() for a in firsts] == classes
    for a, cls in zip(firsts, classes, strict=True):              # each row shows its own class
        row = np.flatnonzero(SHAPES["labels"] == cls)
        assert any(np.allclose(_pixels(a), SHAPES["images"][i] / 255) for i in row)
    assert not axs[0].get_xticks().size
    assert axs[0].images[0].get_interpolation() == "nearest"


def test_imgrid_seed_repeats_the_same_picks() -> None:
    first = [_pixels(a) for a in kx.plot(SHAPES["images"], SHAPES["labels"], "imgrid", seed=3)]
    again = [_pixels(a) for a in kx.plot(SHAPES["images"], SHAPES["labels"], "imgrid", seed=3)]
    other = [_pixels(a) for a in kx.plot(SHAPES["images"], SHAPES["labels"], "imgrid", seed=4)]
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other, strict=True))


def test_imgrid_short_class_leaves_a_short_row() -> None:
    labels = np.array(["a"] * 5 + ["b"] * 2)
    axs = kx.plot(RGB[[0, 1, 2, 3, 4, 0, 1]], labels, "imgrid", n=3)
    assert len(axs) == 3 + 2
    assert len(axs[0].figure.axes) == 5                           # the empty cell is removed


def test_imgrid_without_labels_and_with_idx() -> None:
    plain = kx.plot(SHAPES["images"], "imgrid", n=9)
    assert len(plain) == 9
    assert all(a.get_title() == "" for a in plain)
    gs = plain[0].get_subplotspec().get_gridspec()
    assert (gs.nrows, gs.ncols) == (3, 3)
    picked = [5, 0, 42]
    axs = kx.plot(SHAPES["images"], SHAPES["labels"], "imgrid", idx=picked)
    assert [a.get_title() for a in axs] == [SHAPES["labels"][i] for i in picked]
    assert np.allclose(_pixels(axs[0]), SHAPES["images"][5] / 255)


def test_imgrid_reads_only_the_images_shown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = []
    for i in range(5):
        paths.append(tmp_path / f"{i}.png")
        plt.imsave(paths[-1], RGB[i])
    read = []
    real = plt.imread
    monkeypatch.setattr(kx._image.plt, "imread", lambda p: read.append(p) or real(p))
    kx.plot(paths, "imgrid", idx=[1, 3])
    assert read == [str(paths[1]), str(paths[3])]


def test_imgrid_figure_size_and_grid_cell() -> None:
    axs = kx.plot(SHAPES["images"], SHAPES["labels"], "imgrid", n=5)
    w, h = axs[0].figure.get_size_inches()
    assert w > h * 5 / 7                                          # 5 wide, 7 tall, about 1.4 in each
    left, right = kx.grid(2)
    size = tuple(left.figure.get_size_inches())
    cell = kx.plot(SHAPES["images"], "imgrid", n=4, ax=right)
    assert tuple(left.figure.get_size_inches()) == size
    assert all(a.get_subplotspec().get_topmost_subplotspec() == right.get_subplotspec() for a in cell)


@pytest.mark.parametrize(("call", "match"), [
    (lambda: kx.plot(RGB, ["a", "b"], "imgrid"), "5 images but 2 labels"),
    (lambda: kx.plot(RGB, "imgrid", idx=[7]), r"idx= \[7\] is out of range"),
    (lambda: kx.plot(RGB, list("abcde"), list("abcde"), "imgrid"), "no z"),
    (lambda: kx.plot(np.zeros((21, 2, 2)), [str(i) for i in range(21)], "imgrid"), "21 classes are too many"),
])
def test_imgrid_errors(call: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


# ---------- meanimg ----------
def test_resize_picks_the_pixel_each_new_one_falls_in() -> None:
    image = np.arange(12.0).reshape(3, 4)
    assert kx._image.resize(image, (6, 2)).tolist() == [[0, 2], [0, 2], [4, 6], [4, 6], [8, 10], [8, 10]]
    assert kx._image.resize(RGB[0], (8, 6)).shape == (8, 6, 3)


def test_moments_sum_once_and_resize_the_sums() -> None:
    small = np.full((2, 2), 0.5)
    big = np.zeros((4, 4))
    stats, shape = kx._image.moments([big, big, small], [np.arange(3)])
    assert shape == (4, 4)                                         # most images are 4 x 4
    mean, squares, n = stats[0]
    assert n == 3
    assert np.allclose(mean, 0.5 / 3)
    assert np.allclose(squares, 0.25 / 3)
    stats, _ = kx._image.moments([np.zeros((2, 2)), np.ones((2, 2, 3))], [np.arange(2)])
    assert stats[0][0].shape == (2, 2, 3)                          # grey joins colour as colour


def test_meanimg_is_the_mean_of_each_class() -> None:
    axs = kx.plot(SHAPES["images"], SHAPES["labels"], "meanimg")
    classes = sorted(set(SHAPES["labels"]))
    assert len(axs) == len(classes)
    for a, cls in zip(axs, classes, strict=True):
        members = SHAPES["images"][SHAPES["labels"] == cls] / 255
        assert np.allclose(_pixels(a), members.mean(axis=0))
        assert a.get_title() == f"{cls} (n={len(members)})"


def test_meanimg_std_maps_share_one_scale_and_colorbar() -> None:
    axs = kx.plot(SHAPES["images"], SHAPES["labels"], "meanimg-std")
    members = SHAPES["images"][SHAPES["labels"] == "blob"] / 255
    assert np.allclose(_pixels(axs[0]), members.std(axis=0).mean(axis=2))
    assert len({a.images[0].get_clim() for a in axs}) == 1
    assert axs[0].images[0].get_clim()[0] == 0
    bars = [a for a in axs[0].figure.axes if a not in axs]
    assert len(bars) == 1
    assert bars[0].get_ylabel() == "std"


def test_meanimg_without_labels_and_wrapping() -> None:
    (only,) = kx.plot(RGB, "meanimg")
    assert only.get_title() == "all (n=5)"
    assert np.allclose(_pixels(only), RGB.mean(axis=0) / 255)
    axs = kx.plot(SHAPES["images"], SHAPES["labels"], "meanimg", cols=3)
    gs = axs[0].get_subplotspec().get_gridspec()
    assert (gs.nrows, gs.ncols) == (3, 3)


def test_meanimg_errors() -> None:
    with pytest.raises(ValueError, match="5 images but 2 labels"):
        kx.plot(RGB, ["a", "b"], "meanimg")
    with pytest.raises(ValueError, match="'meanimg' takes the images"):
        kx.plot(RGB, list("abcde"), list("abcde"), "meanimg")


# ---------- colorhist ----------
def _hist(image: np.ndarray, bins: int = 32) -> np.ndarray:
    return np.histogram(image, bins=bins, range=(0, 1))[0] / image.size


def test_histograms_average_each_images_shares() -> None:
    big, small = np.zeros((4, 4)), np.ones((2, 2))
    hists, channels = kx._image.histograms([big, small], [np.arange(2)], 4, hsv=False)
    assert channels == ["grey"]
    assert hists[0, 0].tolist() == [0.5, 0, 0, 0.5]               # each image weighs the same, whatever its size
    hists, channels = kx._image.histograms([big, np.ones((2, 2, 3))], [np.arange(2)], 4, hsv=False)
    assert channels == ["red", "green", "blue"]
    assert all(h.tolist() == [0.5, 0, 0, 0.5] for h in hists[0])  # grey among colour: R = G = B


def test_colorhist_one_panel_per_channel_one_line_per_class() -> None:
    axs = kx.plot(SHAPES["images"], SHAPES["labels"], "colorhist")
    classes = sorted(set(SHAPES["labels"]))
    assert [a.get_title() for a in axs] == ["red", "green", "blue"]
    assert [len(a.lines) for a in axs] == [len(classes)] * 3
    members = SHAPES["images"][SHAPES["labels"] == classes[0]] / 255
    want = np.mean([_hist(m[..., 1]) for m in members], axis=0)
    assert np.allclose(axs[1].lines[0].get_ydata(), want)
    assert all(a.get_ylim() == axs[0].get_ylim() for a in axs)
    assert axs[0].get_ylim()[1] >= max(ln.get_ydata().max() for a in axs for ln in a.lines)   # no peak cut
    assert [t.get_text() for t in axs[-1].get_legend().get_texts()] == classes
    styles = [ln.get_linestyle() for ln in axs[0].lines]
    colours = len(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
    assert styles[:colours] == ["-"] * min(colours, len(classes))  # dashed only once the colours run out


def test_colorhist_hsv_and_map() -> None:
    axs = kx.plot(SHAPES["images"], SHAPES["labels"], "colorhist-hsv-map", bins=16)
    assert [a.get_title() for a in axs] == ["hue", "saturation", "value"]
    shown = axs[2].images[0].get_array()
    assert shown.shape == (len(set(SHAPES["labels"])), 16)
    assert np.allclose(shown.sum(axis=1), 1)                      # each class's shares add up to all pixels
    assert len({a.images[0].get_clim() for a in axs}) == 1
    assert axs[0].get_yticklabels()[0].get_text() == "blob"


def test_colorhist_grey_images_and_errors() -> None:
    (only,) = kx.plot(RGB[..., 0], "colorhist")
    assert only.get_title() == "grey"
    assert only.get_legend() is None
    with pytest.raises(ValueError, match="at least 2 bins"):
        kx.plot(RGB, "colorhist", bins=1)
    with pytest.raises(ValueError, match="'colorhist' takes the images"):
        kx.plot(RGB, list("abcde"), list("abcde"), "colorhist")


# ---------- imhist ----------
def test_imhist_rows_of_image_and_its_channel_histograms() -> None:
    axs = kx.plot(SHAPES["images"], SHAPES["labels"], "imhist", idx=[4, 9], bins=16)
    assert len(axs) == 4                                          # image, histogram; image, histogram
    image, hist = axs[2], axs[3]
    assert np.allclose(_pixels(image), SHAPES["images"][9] / 255)
    assert image.get_title() == SHAPES["labels"][9]
    assert [ln.get_label() for ln in hist.lines] == ["red", "green", "blue"]
    assert to_hex(hist.lines[0].get_color()) == "#d62728"         # each channel in its own colour
    assert np.allclose(hist.lines[1].get_ydata(), _hist(SHAPES["images"][9][..., 1] / 255, 16))
    assert axs[1].get_shared_x_axes().joined(axs[1], hist)
    assert hist.get_xlabel() == "value"
    assert axs[1].get_legend() is not None


def test_imhist_picks_n_random_images_by_seed() -> None:
    first = kx.plot(SHAPES["images"], "imhist", n=2, seed=5)
    again = kx.plot(SHAPES["images"], "imhist", n=2, seed=5)
    assert len(first) == 4
    assert all(np.array_equal(_pixels(a), _pixels(b)) for a, b in zip(first[::2], again[::2], strict=True))
    assert len(kx.plot(RGB, "imhist")) == 2 * 4                   # 4 by default


def test_imhist_grey_and_hsv() -> None:
    (_, grey) = kx.plot(RGB[..., 0], "imhist", idx=[0])
    assert [ln.get_label() for ln in grey.lines] == ["grey"]
    (_, hsv) = kx.plot(RGB, "imhist-hsv", idx=[0])
    assert [ln.get_label() for ln in hsv.lines] == ["hue", "saturation", "value"]


def test_imhist_errors() -> None:
    with pytest.raises(ValueError, match="5 images but 2 labels"):
        kx.plot(RGB, ["a", "b"], "imhist")
    with pytest.raises(ValueError, match="at least 2 bins"):
        kx.plot(RGB, "imhist", bins=1)
    with pytest.raises(ValueError, match="out of range"):
        kx.plot(RGB, "imhist", idx=[9])
