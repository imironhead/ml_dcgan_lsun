# DCGAN on bedromms dataset of LSUN

In commit 62191804822c5f42456a7075f3db8368b98c4381, the progress looks promising. There are actually no changes. Sometimes the model encounters **"NaN"** (not a number) errors with unknown reasons. I rolled back some steps to bypass that issue. There was an accident so the older records in the chart is lost.

![result after 11500 batches](/assets/batch_11500.png)

![discriminator loss](/assets/discriminator_11500.png)

![generator loss](/assets/generator_11500.png)

---

In commit 8cb54369787640881b4f32ca7556289f58bd987a, got **NaN** after 18900 batches:

![collapse after 18900 batches](/assets/collapse_18900.gif)

---

In commit 7bbc486576a27f1106ced18f5f86bed800ae73c1, got **NaN** after 5400 batches:

![collapse after 5400 batches](/assets/collapse_5400.gif)

---

# Dependencies:

* tensorflow
* numpy
* scipy
* lmdb
