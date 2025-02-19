{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "# Install Kaggle\n",
        "!pip install kaggle --quiet\n",
        "\n",
        "# Mount Google Drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# Copy kaggle.json from Drive (Change the path if needed)\n",
        "!mkdir -p ~/.kaggle\n",
        "!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/\n",
        "!chmod 600 ~/.kaggle/kaggle.json  # Set correct permissions\n",
        "\n",
        "# Download HAM10000 dataset\n",
        "!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000\n",
        "\n",
        "# Unzip the dataset\n",
        "!unzip -q skin-cancer-mnist-ham10000.zip -d skin_cancer_data\n",
        "\n",
        "# Verify dataset extraction\n",
        "import os\n",
        "print(\"Dataset files:\", os.listdir(\"skin_cancer_data\"))\n",
        "\n",
        "# Load metadata\n",
        "import pandas as pd\n",
        "df = pd.read_csv(\"skin_cancer_data/HAM10000_metadata.csv\")\n",
        "print(df.head())  # Display first 5 rows of metadata"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E7c2aGQU_HWG",
        "outputId": "cd859dad-b442-4408-e8a7-cffa811db770"
      },
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n",
            "cp: cannot stat '/content/drive/MyDrive/kaggle.json': No such file or directory\n",
            "chmod: cannot access '/root/.kaggle/kaggle.json': No such file or directory\n",
            "Dataset URL: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000\n",
            "License(s): CC-BY-NC-SA-4.0\n",
            "skin-cancer-mnist-ham10000.zip: Skipping, found more recently modified local copy (use --force to force download)\n",
            "replace skin_cancer_data/HAM10000_images_part_1/ISIC_0024306.jpg? [y]es, [n]o, [A]ll, [N]one, [r]ename: "
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GLCErBSQ-cPw",
        "outputId": "6c8ec676-8240-4695-e639-848eece42b7a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     lesion_id      image_id   dx dx_type   age   sex localization\n",
            "0  HAM_0000118  ISIC_0027419  bkl   histo  80.0  male        scalp\n",
            "1  HAM_0000118  ISIC_0025030  bkl   histo  80.0  male        scalp\n",
            "2  HAM_0002730  ISIC_0026769  bkl   histo  80.0  male        scalp\n",
            "3  HAM_0002730  ISIC_0025661  bkl   histo  80.0  male        scalp\n",
            "4  HAM_0001466  ISIC_0031633  bkl   histo  75.0  male          ear\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  0%|          | 23/10015 [00:00<01:26, 115.15it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031633.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029396.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032417.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031326.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031029.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029836.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032129.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032343.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032128.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  1%|          | 51/10015 [00:00<01:15, 131.74it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030698.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031753.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031159.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031017.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029559.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030661.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031650.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029687.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032013.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031691.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030105.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  1%|          | 84/10015 [00:00<01:09, 142.29it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030377.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031468.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030926.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031008.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031495.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031485.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029413.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029576.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031967.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031584.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029912.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033539.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032283.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030005.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030189.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030768.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029837.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031624.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030607.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  1%|          | 117/10015 [00:00<01:05, 152.10it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029308.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029425.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030565.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032463.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032306.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032304.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031639.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031212.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032382.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029674.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029613.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029418.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  3%|▎         | 274/10015 [00:01<00:22, 438.33it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034318.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032972.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032534.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033785.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033184.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032929.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032963.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032949.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033322.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033127.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033041.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033900.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033361.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032732.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032778.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033466.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030276.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033583.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033465.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029683.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033482.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033201.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032514.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033691.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033910.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032556.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033646.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033592.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033899.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032842.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032618.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032513.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033716.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034011.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033613.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032877.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034175.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033523.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029929.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032978.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033973.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033761.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033505.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033280.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033212.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033770.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032829.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033601.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033284.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033908.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033400.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033677.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033262.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033437.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032688.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033195.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032553.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033252.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032570.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033736.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033378.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033379.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033438.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032983.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034024.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033778.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032863.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034168.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033582.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033576.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033410.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032725.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033952.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033264.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033606.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034103.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032883.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032665.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033130.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033321.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034189.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033306.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032757.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032508.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034201.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032776.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033828.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033235.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034070.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033709.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032826.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034115.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033776.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033056.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033853.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033453.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034126.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033744.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033659.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032567.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033088.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032756.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032654.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032576.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033587.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032675.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032997.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033529.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033581.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033913.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034235.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034007.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034303.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034031.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033124.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033449.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033676.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032898.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033488.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033758.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033397.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033414.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034113.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033507.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033791.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034259.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033446.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033200.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029472.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029682.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033642.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033486.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034003.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033079.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033629.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034037.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032876.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033635.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033868.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034125.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033060.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032099.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032215.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030124.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029600.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031376.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031876.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031620.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030542.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029548.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030705.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030272.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031345.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032063.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032325.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031989.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029394.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r  3%|▎         | 318/10015 [00:01<00:37, 258.77it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032280.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032335.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030118.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030721.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031556.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032168.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031130.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r  4%|▎         | 352/10015 [00:01<00:48, 199.59it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031321.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032486.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029340.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030731.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031542.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031812.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030465.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030706.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029522.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031424.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029770.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030227.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r  4%|▍         | 380/10015 [00:01<00:55, 173.06it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030095.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031394.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031060.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031593.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032348.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032200.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030744.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r  4%|▍         | 403/10015 [00:02<01:01, 156.19it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032156.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030783.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029358.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031766.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030258.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030203.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032051.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032481.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030459.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029776.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029519.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030522.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  4%|▍         | 441/10015 [00:02<01:03, 150.21it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031686.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030208.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032006.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030723.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032373.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029568.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030561.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031591.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030007.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030812.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029802.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031770.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029557.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  5%|▍         | 473/10015 [00:02<01:10, 136.01it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032123.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029789.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030026.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031851.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030436.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031819.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029420.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032230.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029525.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029474.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032045.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r  5%|▍         | 488/10015 [00:02<01:16, 124.03it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031016.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030346.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029991.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029603.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031449.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030649.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032043.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030806.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029791.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  5%|▌         | 514/10015 [00:03<01:25, 111.64it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031168.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030822.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030636.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029897.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029596.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029580.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  5%|▌         | 538/10015 [00:03<01:29, 105.62it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031078.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031464.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032116.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030329.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029311.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030396.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031702.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031061.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r  5%|▌         | 549/10015 [00:03<01:29, 106.19it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029878.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031289.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030172.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031580.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030758.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030365.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  6%|▌         | 570/10015 [00:03<01:50, 85.78it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030188.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029810.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029678.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031937.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030683.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  6%|▌         | 589/10015 [00:03<01:49, 86.06it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029585.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031050.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030876.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029841.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030319.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030965.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029617.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031783.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030137.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  6%|▌         | 613/10015 [00:04<01:37, 96.37it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031808.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030458.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031522.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031352.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031150.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032024.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032235.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031961.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030533.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032456.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  6%|▋         | 636/10015 [00:04<01:32, 101.72it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031716.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031138.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031903.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029384.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031630.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030318.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031459.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032445.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  7%|▋         | 661/10015 [00:04<01:24, 110.48it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030849.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030801.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032358.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031200.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031707.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031537.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030231.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030081.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031334.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030988.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  7%|▋         | 686/10015 [00:04<01:31, 101.54it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031951.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030310.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030608.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029588.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031181.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031287.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032365.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030935.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032040.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030372.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  7%|▋         | 707/10015 [00:05<01:45, 88.30it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029849.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030976.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032303.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032103.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  7%|▋         | 726/10015 [00:05<01:52, 82.40it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029406.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031980.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029764.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030240.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  7%|▋         | 744/10015 [00:05<01:58, 78.47it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029329.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031519.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031261.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029612.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029443.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030700.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031496.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  8%|▊         | 764/10015 [00:05<01:50, 83.83it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031465.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031831.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030241.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030605.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031000.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030130.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031037.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029627.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031825.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030677.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029927.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  8%|▊         | 786/10015 [00:06<01:45, 87.68it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030383.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032395.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032031.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031888.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031872.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031349.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  8%|▊         | 804/10015 [00:06<01:54, 80.53it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030835.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031218.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029505.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  8%|▊         | 823/10015 [00:06<01:47, 85.42it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030394.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031677.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031277.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031853.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031329.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031528.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030369.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031600.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032159.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  9%|▊         | 855/10015 [00:06<01:33, 97.72it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029527.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032111.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031132.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031987.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029823.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029455.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030123.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031945.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029931.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029872.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029947.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  9%|▉         | 877/10015 [00:07<01:38, 92.95it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030998.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029320.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029427.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029464.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029924.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031893.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032498.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  9%|▉         | 896/10015 [00:07<01:48, 84.15it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029778.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029555.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029518.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 10%|▉         | 981/10015 [00:07<00:35, 255.62it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030173.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032170.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033460.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033722.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034315.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032967.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033199.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032719.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030630.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033305.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032179.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031428.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032315.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032681.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033509.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032694.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034197.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034057.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033260.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032643.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033732.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032827.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033022.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032835.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033480.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034165.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033185.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033783.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033693.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032804.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033924.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033531.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032636.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033270.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033246.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032843.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033528.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033865.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034296.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032763.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034291.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034167.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033949.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032773.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033156.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033701.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033854.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033884.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034142.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033784.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033307.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033987.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033553.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033401.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033169.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033855.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033660.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034221.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033945.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033491.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034280.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034151.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033391.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033750.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033685.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033631.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033490.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034283.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032740.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034186.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031601.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034252.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033829.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030323.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 10%|█         | 1008/10015 [00:07<00:43, 209.29it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032113.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032330.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032271.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031897.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031253.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031619.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031396.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031196.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032249.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029801.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032124.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029793.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031133.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030959.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 10%|█         | 1031/10015 [00:08<00:53, 166.87it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031511.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030316.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029345.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031226.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032460.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030067.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031436.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030088.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 10%|█         | 1051/10015 [00:08<01:01, 145.42it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031088.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030160.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029731.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031558.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031577.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031024.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030056.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030207.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 11%|█         | 1068/10015 [00:08<01:05, 135.90it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031223.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030226.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031125.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029676.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032359.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029741.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031362.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 11%|█         | 1124/10015 [00:08<00:57, 154.23it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029880.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030488.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031761.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030583.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030789.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031309.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030021.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032468.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029760.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030555.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030244.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031827.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031002.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032410.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030579.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034169.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032941.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032642.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033790.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033675.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030442.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032247.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031429.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030321.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 11%|█▏        | 1143/10015 [00:08<00:55, 159.51it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029967.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032138.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029783.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029578.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030011.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030427.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029891.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030757.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031358.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031735.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032114.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031344.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029962.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031457.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029824.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031372.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031443.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030015.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 12%|█▏        | 1184/10015 [00:09<00:51, 172.83it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031123.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032613.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033256.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034135.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033626.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033780.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033891.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033554.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033810.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033695.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033005.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033422.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033808.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033847.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033860.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030830.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031271.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030665.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031257.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 12%|█▏        | 1219/10015 [00:09<01:07, 130.94it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029973.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031799.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030870.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030623.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031023.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031177.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 12%|█▏        | 1234/10015 [00:09<01:07, 130.81it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030417.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032396.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030443.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031499.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032248.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029512.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030486.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029575.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030119.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030129.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032214.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 13%|█▎        | 1263/10015 [00:09<01:09, 125.49it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030281.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031118.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030080.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031953.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031406.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032079.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031642.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031711.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 13%|█▎        | 1291/10015 [00:10<01:10, 124.46it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032331.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032400.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033186.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030060.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030211.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030171.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029538.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032313.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029913.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032265.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032308.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030134.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030445.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030798.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032019.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031557.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030898.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032244.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032098.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 13%|█▎        | 1327/10015 [00:10<01:01, 141.15it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029651.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032044.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030759.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032450.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032046.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029630.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031369.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029547.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031941.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030818.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 14%|█▎        | 1356/10015 [00:10<01:06, 130.96it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030106.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031779.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031745.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031586.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031972.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030539.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031977.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031517.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029370.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031958.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030521.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030089.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 14%|█▍        | 1383/10015 [00:10<01:11, 119.94it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031598.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029473.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030014.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029571.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032220.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030165.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031189.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032190.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029558.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031890.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031545.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 14%|█▍        | 1424/10015 [00:11<01:08, 125.21it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033312.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029884.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030223.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032182.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030760.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034006.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029660.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029343.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030552.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029726.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029513.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 14%|█▍        | 1450/10015 [00:11<01:11, 120.17it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030824.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031310.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032408.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030440.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031013.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032441.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030324.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029434.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 15%|█▌        | 1509/10015 [00:11<00:41, 206.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033496.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032927.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030681.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029562.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029453.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034071.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033573.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030828.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032847.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033654.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030575.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033931.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033268.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032750.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032888.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033968.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033333.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033717.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034120.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032532.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033500.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033834.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033678.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034046.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032550.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033700.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032604.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033428.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033975.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033278.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033679.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034141.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033643.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032629.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033152.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033636.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029909.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033432.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032833.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034236.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033027.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034118.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033476.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033047.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 16%|█▌        | 1597/10015 [00:11<00:22, 381.01it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032017.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033946.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033369.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032447.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033655.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033831.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032583.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034162.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032187.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032976.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033274.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033034.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034294.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034202.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033416.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032662.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033452.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033074.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034087.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032722.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033204.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033953.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032823.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032733.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033183.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032564.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034140.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034256.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032984.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033362.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033947.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033881.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029363.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033820.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033814.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032875.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033802.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033429.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032685.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032903.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033323.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032637.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032993.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033804.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033801.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033216.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033154.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033192.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034222.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032569.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034064.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033902.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032774.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032840.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034313.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033570.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032602.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032724.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033129.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033180.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033962.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033603.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033325.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034208.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033068.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034218.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034263.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034262.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033103.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034106.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033725.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032537.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033287.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033286.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032586.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029632.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033485.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033175.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033886.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032892.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033942.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033473.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030564.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033337.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033594.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033275.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033938.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032841.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033819.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032544.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032844.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032921.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034117.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033415.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033240.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032782.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032522.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033595.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034059.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033063.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033174.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033239.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033985.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032726.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033272.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032718.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032736.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033317.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034269.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033956.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033241.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032598.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034289.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033980.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032533.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033986.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033004.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032072.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033578.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034100.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032940.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032511.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033687.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034085.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033299.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033479.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033522.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033911.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033779.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032638.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032603.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033871.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032955.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032812.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033619.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034065.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034132.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033038.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033630.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032879.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034012.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034150.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033209.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034074.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032699.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033848.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032622.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034246.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032915.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 17%|█▋        | 1752/10015 [00:11<00:15, 545.75it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033883.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034145.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032981.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033470.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034094.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033560.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034061.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032716.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034159.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033477.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034092.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033863.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034180.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033267.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034005.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033872.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032617.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032965.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034242.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033117.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034104.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032547.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033644.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033806.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032913.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033651.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033304.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032806.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033624.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033238.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034089.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034048.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033905.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033981.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033336.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033405.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034188.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033269.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033569.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032535.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033042.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032081.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034034.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033217.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033728.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033901.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032917.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032925.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032797.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032596.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032856.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033193.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033125.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032810.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033324.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034316.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033710.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033469.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033663.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034284.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031709.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032517.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032684.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034052.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033171.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034022.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033017.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033874.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033754.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032938.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032610.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032958.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033538.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032836.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033444.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034183.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033426.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033546.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032869.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033524.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034101.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030333.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032695.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033310.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034253.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033120.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033081.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034211.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033545.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033141.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033261.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032766.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030031.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032531.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032970.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030391.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032845.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034036.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030238.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032968.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 19%|█▊        | 1873/10015 [00:12<00:15, 527.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032462.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033704.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034239.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034287.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031479.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033166.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033320.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032526.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033813.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032624.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033977.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034002.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033387.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032552.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033279.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032887.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034205.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033612.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034049.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033300.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033616.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029606.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033099.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032723.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032751.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032597.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032982.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033196.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032653.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032730.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032709.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033206.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033518.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033843.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033344.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032988.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033812.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033653.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032698.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033611.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033586.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033226.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033198.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032790.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033245.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033029.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030883.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033670.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029740.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033593.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032509.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034068.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033392.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032873.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034107.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033368.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032630.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033399.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032656.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032592.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033024.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034134.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033878.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032687.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033055.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033559.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032922.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032672.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034076.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033520.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034028.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033487.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033607.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033568.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034051.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033377.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033957.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034172.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033393.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033534.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032626.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034062.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034170.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033061.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033967.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033051.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033050.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033526.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032862.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033073.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033696.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033995.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033668.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032690.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032918.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034000.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032807.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032645.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033498.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031784.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034275.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033440.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033807.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033533.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034173.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033454.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033893.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032960.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033708.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034317.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031741.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033662.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033037.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033773.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033752.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033424.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034216.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032152.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033730.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032975.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034243.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034050.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033114.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032559.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030150.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033394.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034233.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033122.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033713.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 19%|█▉        | 1930/10015 [00:12<00:15, 530.94it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033638.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033805.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033302.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033925.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033431.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033089.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033562.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031233.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033885.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032987.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032872.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033178.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033023.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031186.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033258.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032870.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033999.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033856.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029502.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029933.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031208.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032424.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031554.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032466.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029914.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032110.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 20%|█▉        | 1984/10015 [00:12<00:33, 240.97it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031795.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031408.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031034.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031666.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029819.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032197.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032134.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030390.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030756.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030187.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030970.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030754.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030034.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029397.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029316.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030356.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030192.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029859.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031077.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031999.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030366.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032245.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 20%|██        | 2025/10015 [00:13<00:39, 201.63it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031386.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031565.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030512.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031718.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029591.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030932.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032036.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031543.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030107.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029698.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029937.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030360.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029843.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029454.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 21%|██        | 2058/10015 [00:13<00:43, 182.46it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030501.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030382.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030689.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032420.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029744.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032425.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 21%|██        | 2107/10015 [00:13<00:50, 157.66it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030110.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031494.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029319.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032219.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029318.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032070.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029839.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030747.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030784.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031146.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 21%|██        | 2127/10015 [00:13<00:52, 149.23it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031239.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031529.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030843.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029780.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030616.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032192.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032372.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031915.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030047.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030910.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 22%|██▏       | 2164/10015 [00:14<00:52, 149.73it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031670.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031339.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032430.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029958.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032476.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031170.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030423.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029353.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032504.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029995.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031908.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030653.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030950.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029885.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 22%|██▏       | 2198/10015 [00:14<00:53, 146.66it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031550.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029642.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029893.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031417.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030695.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031203.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030183.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031401.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030255.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029495.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032269.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031350.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029574.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030246.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030032.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 22%|██▏       | 2230/10015 [00:14<00:57, 134.52it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031561.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031410.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029705.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032207.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032149.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032433.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031931.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030083.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030002.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032204.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032095.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031821.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030795.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 23%|██▎       | 2259/10015 [00:14<01:03, 122.28it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029480.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032232.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031005.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030507.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031498.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032287.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030901.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030995.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031368.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031900.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029570.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 23%|██▎       | 2296/10015 [00:15<00:53, 145.53it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031389.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030925.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030122.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031778.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031377.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030604.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031025.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030951.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031957.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031303.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031087.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029652.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031295.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 23%|██▎       | 2330/10015 [00:15<00:56, 136.08it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030929.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032389.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031746.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030277.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031844.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031197.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031270.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029486.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031901.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032076.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029404.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030275.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031217.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 24%|██▍       | 2381/10015 [00:15<00:39, 195.21it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029448.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030956.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030770.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032240.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031759.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029889.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030882.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029514.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032745.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032557.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033092.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033450.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033254.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033158.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032867.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032545.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032692.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033608.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033969.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033031.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032775.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033458.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032614.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033349.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032919.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032866.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033135.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032538.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033591.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031648.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033565.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033123.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033230.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033762.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034214.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033749.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034196.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031090.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032270.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031950.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 24%|██▍       | 2402/10015 [00:15<00:44, 169.93it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030722.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030606.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031346.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031719.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030070.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031093.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029608.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032715.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033991.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033817.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032839.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032932.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031996.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032890.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033844.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029439.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031201.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030283.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031706.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032409.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 24%|██▍       | 2439/10015 [00:16<00:53, 141.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029742.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029877.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031215.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030104.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031103.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031065.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031955.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 25%|██▍       | 2469/10015 [00:16<01:00, 124.88it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031276.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032057.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034093.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031513.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032384.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 25%|██▍       | 2497/10015 [00:16<01:05, 115.44it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029755.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030574.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031585.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030446.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031527.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031166.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030893.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030261.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032696.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029466.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 25%|██▌       | 2530/10015 [00:16<00:56, 132.59it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031400.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029747.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030778.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031749.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029847.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031943.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030335.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029647.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029515.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030271.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031140.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032139.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031243.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030687.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030766.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030800.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 26%|██▌       | 2562/10015 [00:17<00:53, 138.95it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029341.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031762.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031442.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032415.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030177.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031284.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031789.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032414.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031824.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030145.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031640.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029899.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030659.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 26%|██▌       | 2591/10015 [00:17<00:58, 126.18it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031272.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030010.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032482.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029779.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029669.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031169.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030349.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034160.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031489.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032660.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033504.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 26%|██▋       | 2647/10015 [00:17<00:38, 190.60it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030114.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032959.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034306.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032799.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033020.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032768.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033205.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033271.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033575.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032894.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033720.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034255.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032834.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034058.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033001.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033354.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032857.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032906.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033579.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034095.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034143.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032652.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033551.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033571.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033666.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033218.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033054.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033257.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034276.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034119.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034047.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034299.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032808.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033483.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032536.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034161.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033747.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030314.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033012.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033468.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033609.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032777.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033019.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033203.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033366.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032816.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034015.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034223.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032611.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033499.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033372.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034066.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034123.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034026.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032741.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 27%|██▋       | 2672/10015 [00:17<00:35, 205.58it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033301.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034155.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033146.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033248.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032850.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030712.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032727.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032991.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033979.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031836.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031569.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029564.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031225.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031450.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031026.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032146.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031539.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030233.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 27%|██▋       | 2693/10015 [00:17<00:40, 179.72it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029331.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029546.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029392.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031698.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031245.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031351.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031413.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032497.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031643.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030644.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 27%|██▋       | 2728/10015 [00:18<01:03, 114.46it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030737.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031384.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031712.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032439.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029391.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030403.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 27%|██▋       | 2754/10015 [00:18<01:06, 108.41it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029805.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029412.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030528.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030511.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030594.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031236.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030813.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 28%|██▊       | 2766/10015 [00:18<01:06, 108.60it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032174.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031266.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032194.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029489.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029974.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031597.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 28%|██▊       | 2790/10015 [00:18<01:10, 102.81it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032307.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032022.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029539.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031258.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030197.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029655.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031175.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030755.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031976.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031122.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032225.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029856.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031007.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 28%|██▊       | 2821/10015 [00:19<00:59, 121.55it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032290.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030230.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031728.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031526.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032461.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030690.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030181.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030964.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031009.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030452.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031925.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031095.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032246.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031057.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 29%|██▊       | 2860/10015 [00:19<01:00, 118.86it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029524.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031294.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031154.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031056.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029820.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029644.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031063.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032314.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029323.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031520.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031171.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031552.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030526.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 29%|██▉       | 2886/10015 [00:19<01:01, 115.61it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029337.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030096.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029831.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030514.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031407.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032030.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031651.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029919.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031378.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030868.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 29%|██▉       | 2915/10015 [00:19<00:57, 124.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029372.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030782.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030915.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031318.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031470.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032092.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029602.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030954.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031062.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030094.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030352.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029745.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030746.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031986.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 29%|██▉       | 2944/10015 [00:20<00:54, 130.43it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029342.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029680.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030339.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032132.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029545.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031298.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030249.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032302.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029352.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031697.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029917.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029501.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030138.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032185.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032222.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032429.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029951.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031614.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 30%|██▉       | 2971/10015 [00:20<00:55, 126.00it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029857.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031263.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032164.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031041.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032316.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032318.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030767.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032061.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031971.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031721.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031531.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 30%|██▉       | 2984/10015 [00:20<00:59, 118.26it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032212.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029828.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031588.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031325.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032115.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031503.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032446.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032285.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029979.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029624.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 30%|███       | 3007/10015 [00:20<01:10, 99.46it/s] "
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031282.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029961.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032226.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031536.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029594.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031575.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032060.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029347.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 30%|███       | 3028/10015 [00:20<01:21, 86.25it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030435.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031184.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031269.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030035.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031403.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 30%|███       | 3050/10015 [00:21<01:14, 93.87it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030993.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031068.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030791.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031463.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031974.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030517.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031455.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029569.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029351.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030102.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031486.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029526.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031895.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030072.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 31%|███       | 3080/10015 [00:21<01:02, 111.08it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031935.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030820.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032210.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031845.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030148.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029540.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032016.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032048.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030013.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031371.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030523.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031100.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031671.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029637.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 31%|███       | 3103/10015 [00:21<01:08, 101.35it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031207.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030918.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031926.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031652.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030846.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029829.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031297.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032391.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032169.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029870.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 31%|███       | 3128/10015 [00:21<01:04, 107.40it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031833.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031164.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029998.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031579.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032317.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029997.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029685.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030338.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030907.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030428.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029373.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030421.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029911.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030682.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 31%|███▏      | 3139/10015 [00:22<01:14, 91.90it/s] "
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031332.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031966.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032501.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029888.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 32%|███▏      | 3159/10015 [00:22<01:23, 81.78it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032087.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032387.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031604.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030617.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029496.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 32%|███▏      | 3177/10015 [00:22<01:26, 79.46it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031331.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031997.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030669.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032490.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030642.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030050.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030573.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 32%|███▏      | 3194/10015 [00:22<01:35, 71.26it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029966.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031106.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 32%|███▏      | 3202/10015 [00:22<01:37, 70.14it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032434.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030016.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030287.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030888.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030599.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032153.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 32%|███▏      | 3222/10015 [00:23<01:27, 77.47it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029523.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030541.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032211.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031388.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 32%|███▏      | 3238/10015 [00:23<01:28, 76.34it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029336.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031653.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030128.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032100.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031632.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031453.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030214.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 33%|███▎      | 3258/10015 [00:23<01:24, 79.70it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031099.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031773.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031881.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029892.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029943.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029561.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 33%|███▎      | 3289/10015 [00:23<01:09, 97.04it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030550.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031704.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030235.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031701.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031185.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029463.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029815.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030968.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032025.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031157.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031234.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031907.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032254.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030903.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 33%|███▎      | 3314/10015 [00:24<01:01, 109.53it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030292.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031730.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030469.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030170.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030322.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030885.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031638.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031330.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032011.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030540.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031151.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030308.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030470.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032366.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 33%|███▎      | 3341/10015 [00:24<00:55, 121.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030936.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030168.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031229.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031777.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030729.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030457.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029729.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032442.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 34%|███▎      | 3367/10015 [00:24<00:59, 112.26it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030302.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031606.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031422.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032453.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029784.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029633.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030880.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031174.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031319.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030376.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029822.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031776.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029385.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 34%|███▍      | 3404/10015 [00:24<01:00, 108.93it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029918.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031237.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029876.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029701.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030536.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030902.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 34%|███▍      | 3432/10015 [00:25<00:53, 122.34it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031837.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030774.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030041.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029317.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030797.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031690.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029619.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030906.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031291.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031947.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030904.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031938.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029386.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029410.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029675.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029751.jpg"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 34%|███▍      | 3454/10015 [00:25<00:44, 148.66it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030293.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031035.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029604.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031771.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031572.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032426.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030411.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030315.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030269.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031571.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032059.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031069.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031219.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031781.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030120.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 35%|███▍      | 3484/10015 [00:25<00:57, 113.51it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031534.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031477.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031153.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030483.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 35%|███▌      | 3513/10015 [00:25<00:52, 124.41it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031995.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029469.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030289.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030836.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029618.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030204.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030003.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032321.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030658.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032193.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030944.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030566.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030400.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030282.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 35%|███▌      | 3542/10015 [00:26<00:48, 132.43it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029703.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031699.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031213.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031038.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032337.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031102.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032231.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029950.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030761.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031402.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031461.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030862.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030049.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032162.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030957.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 36%|███▌      | 3556/10015 [00:26<00:54, 118.70it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030547.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031288.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031392.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031909.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030787.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030000.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031199.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 36%|███▌      | 3580/10015 [00:26<01:11, 90.53it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031250.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031438.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029972.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030519.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 36%|███▌      | 3599/10015 [00:26<01:21, 78.60it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031021.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031469.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032073.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030871.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031582.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 36%|███▌      | 3619/10015 [00:27<01:14, 85.98it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029699.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032228.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032201.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032066.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030955.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031279.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030804.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029628.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032288.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 36%|███▌      | 3628/10015 [00:27<01:21, 77.96it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031051.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031437.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031313.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 36%|███▋      | 3645/10015 [00:27<01:29, 71.02it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030569.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031661.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032419.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031797.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030943.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029848.jpg"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 37%|███▋      | 3669/10015 [00:27<01:09, 91.87it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030736.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031685.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029426.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029335.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030875.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031647.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030351.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029883.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 37%|███▋      | 3709/10015 [00:27<00:55, 112.96it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030206.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029381.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030274.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032133.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031256.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032071.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031114.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030734.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032484.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032367.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 37%|███▋      | 3735/10015 [00:28<00:53, 118.44it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032202.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030332.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029697.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031364.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031963.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031625.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032002.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031800.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030743.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030374.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031398.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029984.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032039.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030306.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 38%|███▊      | 3766/10015 [00:28<00:48, 130.05it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030202.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030911.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032166.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032266.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031682.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031734.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029471.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031173.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030285.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030570.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 38%|███▊      | 3795/10015 [00:28<00:48, 128.32it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030524.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031018.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029708.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030538.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031637.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030092.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031190.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031544.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030814.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030193.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030448.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031504.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031523.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030679.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032336.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030477.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 38%|███▊      | 3828/10015 [00:28<00:44, 139.96it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031882.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031390.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030878.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032467.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029490.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032005.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031086.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029346.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030660.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030461.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030239.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031774.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031176.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031865.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030237.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032091.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030174.jpg"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 39%|███▊      | 3864/10015 [00:29<00:40, 153.60it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030099.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032263.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032241.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029388.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031367.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030596.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030502.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030974.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031231.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030337.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030668.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031636.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031714.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 39%|███▊      | 3880/10015 [00:29<00:40, 153.35it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030481.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032088.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029367.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032223.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031355.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030509.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030407.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032188.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030490.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031172.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029689.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 39%|███▉      | 3911/10015 [00:29<00:42, 142.98it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030748.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031820.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029599.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029436.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032411.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029999.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030711.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031978.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032108.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030971.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031898.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029330.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 39%|███▉      | 3940/10015 [00:29<00:47, 128.81it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032487.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031357.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032507.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031156.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030186.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032234.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029940.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031097.jpg"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 40%|███▉      | 3970/10015 [00:29<00:45, 132.40it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032056.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030493.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030518.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031715.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029492.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029508.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032454.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030484.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031167.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030418.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 40%|███▉      | 4004/10015 [00:30<00:42, 140.46it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031581.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030262.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032399.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029706.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029550.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029639.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032398.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030556.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029798.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030560.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030414.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030972.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032143.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030141.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029536.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 40%|████      | 4034/10015 [00:30<00:42, 141.15it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030815.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030413.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029312.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030545.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030307.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030921.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030657.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030380.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 41%|████      | 4063/10015 [00:30<00:45, 129.43it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029772.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031792.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031656.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030441.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029871.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031532.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030913.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031434.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 41%|████      | 4090/10015 [00:30<00:46, 127.83it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029688.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031320.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030896.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032505.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032080.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029456.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030692.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031055.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032352.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031273.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029748.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 41%|████      | 4120/10015 [00:30<00:43, 136.94it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032361.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030775.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030752.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031508.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032339.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031490.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031675.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030866.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031998.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029484.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031693.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031540.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 41%|████▏     | 4148/10015 [00:31<00:42, 136.85it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029980.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032050.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031033.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029711.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031104.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029695.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031165.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030291.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031785.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032075.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 42%|████▏     | 4176/10015 [00:31<00:43, 134.31it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030402.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029844.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030265.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030505.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032500.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030248.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031144.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030578.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 42%|████▏     | 4203/10015 [00:31<00:50, 115.52it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031724.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032084.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032052.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029440.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029387.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031004.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030373.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032183.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 42%|████▏     | 4233/10015 [00:31<00:46, 124.32it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029322.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030468.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031240.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030037.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029449.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032272.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032320.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032250.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032239.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030781.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 43%|████▎     | 4264/10015 [00:32<00:42, 135.39it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030450.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031969.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029795.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031383.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029721.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030614.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029992.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031116.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031535.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030357.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030513.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030562.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032198.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030643.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030684.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 43%|████▎     | 4296/10015 [00:32<00:39, 143.28it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031162.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030916.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032004.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032413.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030154.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032180.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032083.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029906.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 43%|████▎     | 4341/10015 [00:32<00:41, 136.17it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031275.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032120.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031343.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031964.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029702.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030989.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030131.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030409.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029307.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030196.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029423.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030385.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030166.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031296.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 43%|████▎     | 4355/10015 [00:32<00:47, 119.43it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032049.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029556.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029377.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031775.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031120.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031089.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030613.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 44%|████▍     | 4390/10015 [00:33<00:40, 140.36it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029864.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031141.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031230.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031032.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030554.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031679.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031373.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031854.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029714.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030073.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029517.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 44%|████▍     | 4425/10015 [00:33<00:38, 144.54it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032499.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031635.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030042.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030764.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030718.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031260.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030897.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030837.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029510.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030068.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030947.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030220.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029364.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 44%|████▍     | 4456/10015 [00:33<00:40, 135.97it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030857.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029806.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030863.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029324.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031864.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031423.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031816.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029722.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030975.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032054.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030101.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031763.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030295.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030342.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029542.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029485.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031801.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 45%|████▍     | 4489/10015 [00:33<00:37, 147.42it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029788.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032479.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031471.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031030.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031525.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030480.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029595.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030867.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030928.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031954.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031764.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029419.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 45%|████▌     | 4521/10015 [00:33<00:39, 138.18it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030927.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031348.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029944.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031128.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029968.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030529.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030984.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029874.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030221.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032268.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029378.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 45%|████▌     | 4549/10015 [00:34<00:45, 120.73it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029903.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031870.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031427.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029956.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031782.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029408.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029684.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030769.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030853.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032276.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029816.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031112.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 46%|████▌     | 4562/10015 [00:34<00:50, 108.20it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031391.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032094.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030612.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031505.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030855.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 46%|████▌     | 4596/10015 [00:34<00:55, 97.59it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030495.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029762.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031787.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029758.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031244.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031467.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032196.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030884.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 46%|████▌     | 4607/10015 [00:34<00:56, 95.40it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032213.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031891.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029483.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031655.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032106.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030368.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032370.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032502.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 46%|████▋     | 4645/10015 [00:35<00:48, 110.51it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029383.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031793.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029532.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029723.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029621.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029503.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029458.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032121.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029866.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029921.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032469.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030087.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 47%|████▋     | 4657/10015 [00:35<00:49, 108.35it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029858.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029925.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032137.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031082.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031492.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031939.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 47%|████▋     | 4680/10015 [00:35<00:53, 100.28it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030584.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029399.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030294.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031861.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030701.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029663.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031747.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032117.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032377.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030510.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030163.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031723.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030999.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 47%|████▋     | 4704/10015 [00:35<00:54, 96.73it/s] "
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030022.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031884.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030799.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031928.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030286.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031634.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031664.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030215.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031669.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030861.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 47%|████▋     | 4739/10015 [00:36<00:51, 102.19it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029757.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029521.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032237.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032053.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030327.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030256.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029767.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030379.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029414.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032216.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030213.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 48%|████▊     | 4760/10015 [00:36<00:55, 94.18it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031729.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032093.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032449.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031665.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031135.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 48%|████▊     | 4785/10015 [00:36<00:52, 99.32it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029605.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031767.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029467.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031110.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031587.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029782.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030103.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031370.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029649.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032255.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031990.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031914.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031382.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 48%|████▊     | 4808/10015 [00:36<00:49, 104.17it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032252.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031739.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031227.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029497.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030398.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029710.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031855.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029398.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 48%|████▊     | 4832/10015 [00:37<00:47, 109.15it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032038.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032171.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030250.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030330.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029852.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030869.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030395.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032403.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029949.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030222.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031553.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031235.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029768.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 48%|████▊     | 4854/10015 [00:37<00:51, 100.06it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031878.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030494.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030987.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030527.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030359.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029338.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 49%|████▊     | 4875/10015 [00:37<00:55, 93.14it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032298.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029732.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029868.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031687.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031992.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031810.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029626.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032416.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 49%|████▉     | 4895/10015 [00:37<00:55, 91.91it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029622.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030763.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029587.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031846.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031022.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029478.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032386.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030199.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032275.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032264.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029365.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030823.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029941.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030147.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 49%|████▉     | 4920/10015 [00:38<00:48, 105.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029691.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031551.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029882.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030924.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031862.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029611.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030178.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 49%|████▉     | 4942/10015 [00:38<00:50, 100.44it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031608.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031590.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032064.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030475.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032464.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029379.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032483.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032032.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030624.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 50%|████▉     | 4966/10015 [00:38<00:51, 97.69it/s] "
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029446.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029380.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029593.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030703.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030467.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032503.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029948.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030228.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029433.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 50%|████▉     | 4986/10015 [00:38<00:51, 97.42it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029922.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032062.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030386.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032401.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030252.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030713.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 50%|████▉     | 5006/10015 [00:38<00:56, 88.40it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031262.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031493.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031574.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032473.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031834.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030309.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030895.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031336.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032010.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 50%|█████     | 5034/10015 [00:39<00:45, 110.16it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032224.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031264.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031979.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031680.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029738.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031627.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031828.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030304.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030982.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031419.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031949.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029586.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031840.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032041.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 51%|█████     | 5065/10015 [00:39<00:38, 129.43it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032160.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031491.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032178.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032451.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029625.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030169.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031444.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032452.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030779.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029681.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030113.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030100.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 51%|█████     | 5092/10015 [00:39<00:42, 115.28it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029631.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029754.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031736.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029692.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030572.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030931.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032167.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030084.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029428.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031324.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 51%|█████     | 5120/10015 [00:39<00:38, 125.62it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031514.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031058.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030577.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031538.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030288.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032351.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 51%|█████▏    | 5156/10015 [00:40<00:32, 150.55it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030438.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031155.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030939.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031179.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030212.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031129.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029590.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030043.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029451.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029990.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029648.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030671.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030667.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032127.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030686.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030881.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029375.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029487.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029935.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030506.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031994.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032122.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 52%|█████▏    | 5188/10015 [00:40<00:33, 144.78it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031210.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032286.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030392.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031075.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030631.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029960.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030716.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030476.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031067.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 52%|█████▏    | 5219/10015 [00:40<00:34, 137.80it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031549.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031214.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030482.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031902.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031073.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029325.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031842.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029853.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030834.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031856.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031684.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 52%|█████▏    | 5249/10015 [00:40<00:34, 138.16it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030071.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030471.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030132.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029321.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029709.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032105.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030419.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030474.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031280.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032259.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031274.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030745.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031300.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029769.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031758.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031780.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 53%|█████▎    | 5281/10015 [00:40<00:33, 140.19it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029766.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029616.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030994.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031028.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030454.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029494.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030201.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029421.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030149.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031843.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031252.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030750.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031516.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029934.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 53%|█████▎    | 5311/10015 [00:41<00:35, 133.41it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029506.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030788.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030046.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031847.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031003.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029690.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029981.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030267.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031616.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 53%|█████▎    | 5340/10015 [00:41<00:34, 134.07it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030864.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032295.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030012.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029969.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029361.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030152.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030381.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031607.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031180.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031124.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031683.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 54%|█████▎    | 5383/10015 [00:41<00:35, 129.39it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029718.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030017.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030216.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030808.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030648.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030489.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030879.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030078.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030345.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030601.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 54%|█████▍    | 5416/10015 [00:42<00:34, 134.55it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032085.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031096.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031873.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029629.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030254.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031251.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029957.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032438.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030153.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030551.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029765.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029552.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031788.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 54%|█████▍    | 5434/10015 [00:42<00:31, 143.70it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031397.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031342.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030325.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031548.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031628.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031859.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030001.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029535.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032000.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031066.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031983.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 55%|█████▍    | 5466/10015 [00:42<00:33, 137.30it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029901.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029989.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031425.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029734.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030627.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030217.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031109.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031307.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030992.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030638.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030472.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030983.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032431.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030647.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 55%|█████▍    | 5498/10015 [00:42<00:31, 142.12it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031772.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029964.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029842.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029807.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029444.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030415.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031020.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031641.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031512.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029818.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031439.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031255.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030516.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031696.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031205.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 55%|█████▌    | 5547/10015 [00:42<00:29, 149.46it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032488.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030634.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030185.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029787.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030200.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029607.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030696.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029863.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030727.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030719.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030030.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032493.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032012.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030751.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029442.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 56%|█████▌    | 5581/10015 [00:43<00:30, 146.93it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029923.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032208.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031480.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031717.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031487.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031858.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031568.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029551.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029620.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030917.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029477.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031879.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030637.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032297.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 56%|█████▌    | 5615/10015 [00:43<00:29, 149.33it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032177.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031830.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031098.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030887.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030708.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032390.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032435.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031246.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031899.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029437.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031530.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030320.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 56%|█████▋    | 5646/10015 [00:43<00:30, 141.94it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031765.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031818.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030098.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031283.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029987.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030331.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031415.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029910.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031265.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030500.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030009.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031936.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030563.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031047.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 57%|█████▋    | 5676/10015 [00:43<00:31, 135.70it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031101.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030040.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030576.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032147.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030290.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030234.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031885.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031312.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032323.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 57%|█████▋    | 5706/10015 [00:44<00:31, 137.37it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031849.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032472.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030811.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030062.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029942.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030680.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032229.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032406.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030439.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031341.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030236.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 57%|█████▋    | 5736/10015 [00:44<00:30, 142.63it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030762.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029896.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030568.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032369.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031755.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030052.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030685.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032293.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030433.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030678.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030343.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030055.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029977.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031328.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031182.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 58%|█████▊    | 5767/10015 [00:44<00:31, 133.88it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031737.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029646.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031533.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030749.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032037.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032319.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029800.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031886.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 58%|█████▊    | 5795/10015 [00:44<00:32, 129.22it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031509.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030948.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029327.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030029.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029712.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030691.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029796.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031768.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 58%|█████▊    | 5823/10015 [00:44<00:33, 126.78it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030097.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030393.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032332.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030558.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030028.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030004.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029887.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031316.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031445.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031458.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030633.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029623.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 58%|█████▊    | 5858/10015 [00:45<00:28, 147.76it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031962.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032299.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030018.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031305.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031414.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030650.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031605.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032130.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029565.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029407.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029890.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032393.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 59%|█████▉    | 5891/10015 [00:45<00:27, 149.94it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030259.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031356.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030715.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031850.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032033.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031137.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030720.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029716.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031232.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029753.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029707.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031027.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030363.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031500.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030949.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030872.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030498.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 59%|█████▉    | 5923/10015 [00:45<00:26, 152.65it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029314.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030938.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030559.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029584.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032440.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031054.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029817.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031826.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032448.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029946.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030530.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029965.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030364.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031912.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030946.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031722.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 59%|█████▉    | 5943/10015 [00:45<00:25, 162.17it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029468.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029445.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030622.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031315.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032432.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029376.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029952.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030641.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031224.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031248.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030317.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031393.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029672.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030515.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029939.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030111.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029461.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 60%|█████▉    | 5978/10015 [00:45<00:26, 154.28it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029971.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032097.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032458.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031562.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030662.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029643.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030195.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029429.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030531.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029326.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030735.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030353.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031083.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031478.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031952.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032381.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030266.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032172.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029673.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031052.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030424.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031441.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 60%|██████    | 6018/10015 [00:46<00:24, 166.00it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031804.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030860.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030776.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032480.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032357.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030724.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030025.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030997.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029850.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030024.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 60%|██████    | 6055/10015 [00:46<00:24, 163.55it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030161.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030218.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031877.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031187.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029529.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032394.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030958.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030848.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031860.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029761.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030027.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032256.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030702.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032104.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031010.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030044.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 61%|██████    | 6088/10015 [00:46<00:27, 143.70it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032131.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029424.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030632.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032101.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030412.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030298.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031946.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031732.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030048.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029936.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030854.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029797.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 61%|██████    | 6103/10015 [00:46<00:29, 133.96it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029834.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031626.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030850.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029954.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029737.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030628.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029700.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 61%|██████    | 6132/10015 [00:47<00:30, 129.27it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030405.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030595.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029928.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030651.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031241.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029908.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031968.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029873.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030389.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029743.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031760.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029658.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031740.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031375.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031751.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 62%|██████▏   | 6166/10015 [00:47<00:28, 135.11it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030590.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032346.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029988.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030676.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029635.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030548.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031910.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030437.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031790.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031786.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032491.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031835.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031193.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 62%|██████▏   | 6194/10015 [00:47<00:31, 121.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029832.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031868.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032326.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032383.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031333.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031916.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030934.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032289.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031387.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 62%|██████▏   | 6207/10015 [00:47<00:35, 106.86it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032353.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030090.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029579.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032001.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031072.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030805.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031432.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030780.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030832.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 62%|██████▏   | 6231/10015 [00:47<00:36, 102.50it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031183.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030587.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030162.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030886.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030969.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031911.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030699.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031484.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 63%|██████▎   | 6266/10015 [00:48<00:35, 104.67it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031242.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030224.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031338.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032360.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030136.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030112.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031610.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031267.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029403.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031566.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031006.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029727.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029416.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 63%|██████▎   | 6288/10015 [00:48<00:35, 103.95it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030889.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029777.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030045.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030496.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031084.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030430.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031944.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030670.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 63%|██████▎   | 6309/10015 [00:48<00:39, 93.69it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031161.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032089.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031433.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031576.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029507.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032495.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030347.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029400.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030190.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030571.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032418.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032324.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 63%|██████▎   | 6355/10015 [00:49<00:28, 127.06it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031139.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030431.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029813.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032506.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030503.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029996.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029916.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032261.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030051.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031420.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030656.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029907.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 64%|██████▎   | 6368/10015 [00:49<00:28, 126.12it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032142.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032354.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032465.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030739.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032077.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030611.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 64%|██████▍   | 6393/10015 [00:49<00:31, 113.93it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030619.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030453.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029677.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029601.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032477.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031869.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032350.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030996.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030753.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 64%|██████▍   | 6416/10015 [00:49<00:38, 93.81it/s] "
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031920.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031589.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029481.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030952.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 64%|██████▍   | 6439/10015 [00:49<00:35, 99.36it/s] "
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029356.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030600.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029763.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031841.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029670.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031507.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030157.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030497.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029447.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031497.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031673.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 65%|██████▍   | 6461/10015 [00:50<00:35, 98.77it/s] "
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032118.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031448.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030020.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031152.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030257.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030075.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 65%|██████▍   | 6485/10015 [00:50<00:32, 108.58it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032282.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029750.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031347.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030908.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030725.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031805.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031688.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029739.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030807.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032485.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032119.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030301.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029804.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031731.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031147.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029694.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 65%|██████▌   | 6511/10015 [00:50<00:29, 118.03it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031327.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029415.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032392.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029509.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029799.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031815.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031744.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030728.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029785.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 65%|██████▌   | 6534/10015 [00:50<00:36, 95.75it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032161.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030054.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031361.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031195.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030985.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029333.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031658.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 65%|██████▌   | 6551/10015 [00:50<00:30, 113.64it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030388.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030499.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031921.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029955.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031472.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030726.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032301.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030270.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031365.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031452.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029511.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029411.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 66%|██████▌   | 6574/10015 [00:51<00:36, 94.72it/s] "
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032068.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032296.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031807.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029310.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 66%|██████▌   | 6597/10015 [00:51<00:33, 102.55it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030205.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029350.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030449.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031456.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029395.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029520.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030738.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030146.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032262.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031426.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030831.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 66%|██████▌   | 6619/10015 [00:51<00:33, 99.96it/s] "
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031416.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031689.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029821.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031906.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029686.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031752.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030865.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031254.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030905.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030109.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031136.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 66%|██████▋   | 6641/10015 [00:51<00:33, 100.58it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030108.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030673.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030278.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031446.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032457.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030618.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029774.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 67%|██████▋   | 6666/10015 [00:52<00:29, 111.80it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030777.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029875.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029886.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031892.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031409.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031695.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030478.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032217.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032155.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030891.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029717.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032205.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030838.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029869.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032267.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030164.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032069.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029657.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 67%|██████▋   | 6692/10015 [00:52<00:30, 110.40it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030646.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032475.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032294.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 67%|██████▋   | 6716/10015 [00:52<00:32, 100.18it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030672.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032363.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030980.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030629.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031178.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029534.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030640.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029845.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031617.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 67%|██████▋   | 6738/10015 [00:52<00:32, 102.00it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032376.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031091.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029790.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032023.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031247.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030697.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032428.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032342.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029752.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 67%|██████▋   | 6759/10015 [00:53<00:33, 97.58it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032086.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030273.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031563.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029653.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029374.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032209.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031404.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 68%|██████▊   | 6780/10015 [00:53<00:34, 94.98it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031145.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032985.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031160.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029735.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029985.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 68%|██████▊   | 6805/10015 [00:53<00:30, 104.51it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031474.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029719.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032158.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031678.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030664.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031733.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032334.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030858.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030609.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032257.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031663.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031883.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 68%|██████▊   | 6831/10015 [00:53<00:27, 114.46it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032096.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029366.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032378.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030247.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030426.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030786.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030520.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032341.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029756.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030874.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031629.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029389.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030312.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029430.jpg"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 68%|██████▊   | 6858/10015 [00:53<00:27, 116.68it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031143.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031863.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031515.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029854.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031418.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031705.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031435.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029615.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030432.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031618.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031142.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031059.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030355.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 69%|██████▊   | 6882/10015 [00:54<00:28, 110.75it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030371.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030741.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031555.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031660.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030666.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031293.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031560.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031646.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031385.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032055.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031518.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 69%|██████▉   | 6911/10015 [00:54<00:24, 124.79it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030063.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029475.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029894.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029986.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030064.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031817.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032421.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029409.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032364.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031281.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030829.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030688.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029825.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 69%|██████▉   | 6936/10015 [00:54<00:28, 108.58it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031959.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032078.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032443.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030460.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030793.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031221.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030194.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 69%|██████▉   | 6959/10015 [00:54<00:31, 95.84it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031380.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029543.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030930.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029945.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031720.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029720.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032150.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029725.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031094.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032345.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031676.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032151.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031615.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031592.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029488.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 70%|██████▉   | 6998/10015 [00:55<00:21, 137.48it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030125.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032191.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030464.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031268.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029402.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029357.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032309.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029846.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031887.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032029.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 70%|███████   | 7032/10015 [00:55<00:20, 147.64it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030139.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031713.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032027.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029976.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031001.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030406.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030979.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031674.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029982.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031622.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031080.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030053.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 71%|███████   | 7064/10015 [00:55<00:20, 141.28it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029435.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030334.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030184.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031337.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030473.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030859.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030350.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031440.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031975.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031501.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032910.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 71%|███████   | 7098/10015 [00:55<00:19, 149.84it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029354.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031748.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032242.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033745.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033112.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033380.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030326.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033356.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033998.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030303.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032470.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029826.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029491.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029422.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031158.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029728.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032181.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033859.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031546.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 72%|███████▏  | 7234/10015 [00:56<00:07, 359.21it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033423.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033846.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033315.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032691.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032540.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032573.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032633.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032007.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034298.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033686.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033251.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033402.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034204.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030462.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033971.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032527.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032964.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033347.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032668.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032891.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033555.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033963.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033793.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033363.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033058.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032924.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033712.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032771.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033297.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030845.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032555.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034254.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033852.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033775.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033561.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034225.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032916.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033756.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033823.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032954.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033371.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033348.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033048.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033870.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033032.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034227.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033563.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031613.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029553.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029953.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033276.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034083.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033316.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034079.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033327.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032871.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034290.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033165.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033430.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030851.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033227.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033365.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033767.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033755.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033625.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032362.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032739.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032874.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032590.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033549.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033921.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033959.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032914.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033309.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034267.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033137.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032953.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034110.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033389.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031904.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034234.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033308.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034274.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033331.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034192.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033689.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030546.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033989.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032977.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033664.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033759.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034144.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032992.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032950.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034156.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034136.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033386.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033003.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033253.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033974.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033941.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033919.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033640.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033927.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032664.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033461.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033511.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 73%|███████▎  | 7271/10015 [00:56<00:10, 261.69it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031466.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031048.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031694.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031412.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029656.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029405.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030709.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030765.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029902.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031988.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032082.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030378.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 73%|███████▎  | 7302/10015 [00:56<00:11, 245.60it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029662.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030582.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029636.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032374.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030085.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029348.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031202.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030892.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030429.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030899.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031757.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032009.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032641.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033345.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032711.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033207.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032558.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034146.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033964.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032951.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033652.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033944.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032717.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032651.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033699.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033915.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032784.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033459.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032710.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032767.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033503.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033795.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032679.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032529.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033119.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033850.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032671.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032566.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033059.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033113.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033547.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033172.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033326.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032743.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033292.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033610.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033100.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033388.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033009.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032928.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032666.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033157.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032811.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034131.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032901.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033370.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033521.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032623.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032821.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032783.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034139.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034133.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034042.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032595.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034281.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034152.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033739.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034020.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033727.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033827.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033497.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034308.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032957.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033334.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034273.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032805.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032753.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033706.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033097.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034014.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034320.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032549.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033633.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034199.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032998.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032902.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032948.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034264.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033657.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034038.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033070.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033837.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032612.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034039.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032785.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032563.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033071.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034086.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032577.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033753.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033242.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032619.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032742.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032961.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034112.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033296.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033838.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033409.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033990.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032937.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032585.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032899.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034157.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033231.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033035.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032525.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033233.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032770.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034184.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033385.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 75%|███████▌  | 7523/10015 [00:56<00:04, 576.35it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033471.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032831.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032780.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033849.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032861.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033984.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032640.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032568.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033126.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034099.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032599.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033472.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032667.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033648.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032825.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032803.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033798.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033014.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032516.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032520.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032832.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033917.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032815.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033094.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033162.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033475.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033210.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033219.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032885.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033291.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034102.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033988.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032859.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032795.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034105.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033138.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032728.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034319.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033748.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034044.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032864.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033763.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033650.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033992.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033715.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033920.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033825.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033857.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032731.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033069.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032735.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033840.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033632.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034025.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033182.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034257.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032999.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032669.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033236.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032802.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032551.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033544.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032749.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034091.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033738.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033510.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033222.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032546.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032521.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032674.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033958.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032989.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033618.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033016.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033906.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033298.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034210.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033351.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033131.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033090.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033832.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032893.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032683.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034111.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032676.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034077.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033641.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033527.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032895.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033091.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034138.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032971.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032781.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033237.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032747.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034286.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033373.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033170.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032680.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033674.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032542.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034224.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033411.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034096.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033382.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034288.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033420.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032661.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032587.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033824.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033830.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032838.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034230.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033057.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034300.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033155.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032609.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033729.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034307.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034137.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032761.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033160.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033930.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033096.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033972.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033367.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033904.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033342.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033265.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032704.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033289.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032822.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033283.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033359.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032905.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034001.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032560.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033427.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033948.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033680.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033085.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033976.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034030.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034179.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034206.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032980.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034314.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034174.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033983.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033018.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032541.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032515.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033492.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032548.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033106.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033506.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033600.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033714.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034190.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033605.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033961.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033702.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033318.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032615.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033862.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032942.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 77%|███████▋  | 7710/10015 [00:56<00:03, 748.25it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032628.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033110.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032701.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034153.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034017.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034215.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033743.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033833.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032512.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034149.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033800.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033163.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033887.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033145.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033799.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032646.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033627.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033803.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032860.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034292.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034207.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033408.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033621.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034116.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034073.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032702.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033667.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033107.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032882.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034078.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032818.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034060.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033002.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033683.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032524.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033376.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033796.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033121.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034158.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033213.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033897.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033694.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034013.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032705.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034198.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033658.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033046.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032689.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034088.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034268.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032908.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034203.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033311.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032700.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032608.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032746.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033266.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033188.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033067.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032787.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033936.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034121.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034232.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034282.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033357.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034178.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033517.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033513.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033649.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033404.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033478.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032707.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034311.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032786.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033788.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033150.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033698.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034240.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033220.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032817.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032820.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032853.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034148.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034241.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033665.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032813.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033558.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033590.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033303.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032584.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033731.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032644.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033637.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033566.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033263.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033815.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034194.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032969.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033894.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033933.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033447.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033639.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033282.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032729.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033076.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033194.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034258.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034023.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033243.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032648.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032620.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030420.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033671.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033711.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034084.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033273.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034176.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033104.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032579.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033189.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032693.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032734.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032659.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032478.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032765.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033128.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033115.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032625.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033867.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033202.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032794.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032801.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032974.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033746.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033384.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032830.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030126.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032530.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033249.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032650.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034182.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034213.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032561.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033839.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032858.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033965.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031814.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032601.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032878.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033093.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033501.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032677.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032769.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033922.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033108.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033940.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033818.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034029.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032649.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034109.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033439.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033214.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033255.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033229.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032909.jpg"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 79%|███████▉  | 7914/10015 [00:57<00:02, 813.67it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033134.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034301.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033914.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033978.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033718.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034237.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034032.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033075.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033406.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033493.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033007.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034010.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032819.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033451.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033552.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034310.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032995.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032754.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032714.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033083.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033133.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033352.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032574.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033398.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033244.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032582.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033052.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032990.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033181.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033661.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034004.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033645.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033530.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032994.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033604.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033634.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034226.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033116.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033136.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033215.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033412.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033514.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033876.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032737.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033339.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031482.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032944.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033516.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032639.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033525.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033519.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032851.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033080.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034129.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033010.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034293.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033350.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033943.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033434.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033688.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033557.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033013.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032809.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032621.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033328.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033768.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033006.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032518.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033548.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033433.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032934.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033742.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034127.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033111.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033142.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032703.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033168.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033077.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033951.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033149.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033343.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033502.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034285.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033966.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034244.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032855.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032792.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034187.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033179.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034251.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033721.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034295.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033929.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033044.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033787.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033599.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033764.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033934.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032565.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033512.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032788.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033221.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033340.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032920.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033673.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033737.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033932.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033062.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034261.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032758.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033939.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032772.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034041.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033598.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034021.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034220.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032973.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033425.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033443.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032593.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033935.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033288.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033087.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033669.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033789.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033419.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032338.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034067.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031290.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033224.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033360.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033294.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033030.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032657.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032605.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032779.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033950.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032886.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032581.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032865.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033726.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033167.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033937.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033703.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034245.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032708.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033574.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032846.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033816.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033672.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033435.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034016.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033960.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033259.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033403.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033954.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033771.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034035.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033880.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033994.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032755.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033724.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033786.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034278.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033102.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032494.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032760.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034128.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034271.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033564.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032627.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033845.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032884.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033532.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032713.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033541.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032670.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033441.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034009.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034305.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034097.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033707.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033912.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032380.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 80%|███████▉  | 8000/10015 [00:57<00:02, 775.45it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032663.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032962.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034272.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032911.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033792.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030492.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033926.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033021.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033835.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034122.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033281.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033285.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033781.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034247.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033455.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034056.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034069.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032673.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033684.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033148.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033109.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032986.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033909.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033293.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033026.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033774.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034191.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033719.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033442.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032943.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033723.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033396.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034231.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034033.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032752.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033794.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033462.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033191.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033508.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033875.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034200.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032539.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032686.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034249.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032881.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033329.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034181.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033277.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033197.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032824.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034209.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034297.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034171.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033996.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033448.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034304.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033036.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033015.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033928.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034124.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033861.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033955.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033822.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032762.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032594.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033147.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033407.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033734.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033923.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033769.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032904.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030447.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034108.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033740.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032519.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034090.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032896.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034166.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034217.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032952.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033851.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032966.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033864.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034008.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033250.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033177.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033086.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032720.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032946.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033153.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033314.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033580.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033993.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032926.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033049.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033463.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033757.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032933.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034063.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033045.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033873.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033143.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032588.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032510.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032798.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032848.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033697.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033656.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034164.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033228.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032744.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033033.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033390.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033836.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033585.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033319.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029393.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029306.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030537.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029654.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030023.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029531.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030961.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030626.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031163.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 81%|████████  | 8081/10015 [00:57<00:03, 534.09it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032322.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031053.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032496.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029775.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030156.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031411.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031848.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031970.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032375.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030362.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031460.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031948.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030543.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031867.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029581.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029814.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031481.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029704.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029809.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030771.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029904.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030284.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029730.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031206.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030361.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031796.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031923.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 81%|████████▏ | 8147/10015 [00:58<00:07, 254.79it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030305.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030232.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030243.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032058.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030504.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030841.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031654.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029334.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030074.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031703.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031645.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031726.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030455.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029431.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030264.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030717.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031521.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 82%|████████▏ | 8196/10015 [00:58<00:09, 189.15it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032021.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030773.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032260.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029664.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030116.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029572.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030933.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030268.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029499.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031956.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032347.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032407.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030840.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032291.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030615.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031917.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031308.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 82%|████████▏ | 8233/10015 [00:59<00:10, 164.48it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031079.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030225.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029313.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030508.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031194.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030597.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029667.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029566.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030058.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 82%|████████▏ | 8262/10015 [00:59<00:11, 148.75it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030990.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029359.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029746.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029450.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031149.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031360.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029861.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032368.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029504.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032492.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 83%|████████▎ | 8286/10015 [00:59<00:11, 144.67it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029835.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031447.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031667.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031839.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031502.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031905.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030367.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032227.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031657.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030077.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029865.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032003.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030466.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030135.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 83%|████████▎ | 8326/10015 [00:59<00:11, 145.34it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030941.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029938.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029516.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030066.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031631.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030588.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029640.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029724.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030057.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032165.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029666.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 83%|████████▎ | 8360/10015 [01:00<00:11, 139.15it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031541.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030852.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030652.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031204.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030179.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030894.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030610.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031340.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030065.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029382.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030093.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 84%|████████▍ | 8394/10015 [01:00<00:11, 146.07it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031857.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031621.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031822.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030654.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031811.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031984.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030945.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029609.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029661.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031623.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029493.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031559.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029482.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 84%|████████▍ | 8410/10015 [01:00<00:11, 138.41it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032305.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031756.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030127.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031725.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030397.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032427.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032355.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031405.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031092.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 85%|████████▍ | 8464/10015 [01:00<00:09, 161.79it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029339.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032184.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031399.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030621.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031510.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031727.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031483.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032274.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032436.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031603.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029867.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031070.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032176.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029589.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029641.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031564.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032141.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031798.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031015.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 85%|████████▍ | 8507/10015 [01:00<00:08, 184.39it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031354.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032163.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030842.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030967.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030675.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032474.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030819.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032015.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031285.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030532.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029459.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029528.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030581.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029438.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029994.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029926.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031934.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029583.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030399.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032136.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031353.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029693.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 85%|████████▌ | 8545/10015 [01:01<00:08, 170.91it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031932.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030592.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031302.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029457.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030525.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030296.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030121.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031126.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029736.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031311.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030816.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030796.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030790.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030914.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 86%|████████▌ | 8580/10015 [01:01<00:09, 155.37it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032300.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032126.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032067.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032489.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030733.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030544.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030962.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029978.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029577.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 86%|████████▌ | 8616/10015 [01:01<00:08, 166.30it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029993.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031117.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031046.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029349.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032047.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032243.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030328.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030117.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031363.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031220.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029369.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031942.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029452.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 86%|████████▋ | 8653/10015 [01:01<00:08, 166.63it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029773.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032444.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029645.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030416.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030180.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031451.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029544.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031612.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030115.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029898.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 87%|████████▋ | 8687/10015 [01:02<00:08, 156.68it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032112.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031306.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030704.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029597.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030082.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029679.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030006.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029432.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030079.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031596.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029881.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031115.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031919.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 87%|████████▋ | 8721/10015 [01:02<00:08, 159.90it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030039.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030535.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031700.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030209.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031019.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029959.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029696.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029975.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031042.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031049.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030219.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030981.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031710.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 87%|████████▋ | 8757/10015 [01:02<00:08, 155.69it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031708.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032251.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032253.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029665.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030966.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029614.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032189.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032281.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030425.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032145.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 88%|████████▊ | 8794/10015 [01:02<00:07, 166.93it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030973.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030444.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030645.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032292.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032107.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031076.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032218.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031838.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032028.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030253.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029733.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031832.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030340.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031278.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030069.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031924.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030167.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032175.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032327.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 88%|████████▊ | 8861/10015 [01:03<00:05, 201.14it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032423.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030873.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031014.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032936.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030732.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030802.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030742.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031374.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030422.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031681.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030313.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030008.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033232.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032405.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031933.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029905.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031081.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029983.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031662.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031583.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032109.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031031.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031742.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031113.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030792.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032459.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032195.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031813.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030354.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031476.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 89%|████████▊ | 8882/10015 [01:03<00:06, 185.85it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031754.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029803.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029465.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031894.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031594.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029855.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029530.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031960.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029554.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031366.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031644.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031802.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030625.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 89%|████████▉ | 8938/10015 [01:03<00:06, 170.73it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032233.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030251.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032144.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030598.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030740.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030772.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029479.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032278.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030580.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030620.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031871.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031567.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 90%|████████▉ | 8985/10015 [01:03<00:05, 198.53it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030809.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032311.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030920.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030370.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030534.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032284.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031107.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031314.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031249.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031121.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032034.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030451.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031301.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031111.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030833.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032140.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031982.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029833.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031806.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030635.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031991.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030817.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029470.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030674.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032065.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029749.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030922.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031192.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 90%|█████████ | 9026/10015 [01:04<00:05, 178.53it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031074.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030603.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031085.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030260.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031462.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031981.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032279.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030198.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032186.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030923.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032340.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032090.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 90%|█████████ | 9045/10015 [01:04<00:05, 171.95it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030401.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032035.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032412.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031930.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031131.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031473.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029671.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031011.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031134.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030061.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030960.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030311.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031875.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 91%|█████████ | 9100/10015 [01:04<00:05, 169.71it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032125.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032471.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032148.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031889.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030553.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031573.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031238.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029771.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031064.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029332.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031304.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030856.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029476.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030456.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030942.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031913.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031880.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029344.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029812.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030639.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030694.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 91%|█████████▏| 9144/10015 [01:04<00:04, 190.19it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029390.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029441.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032310.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030663.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030151.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030279.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032344.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031866.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030033.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031039.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031611.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032074.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030978.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030810.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 92%|█████████▏| 9184/10015 [01:04<00:04, 179.99it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030839.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029862.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030593.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031045.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031794.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030176.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031379.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031599.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031965.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030140.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030159.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030847.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032157.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030434.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032236.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030919.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 92%|█████████▏| 9203/10015 [01:05<00:04, 165.73it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030210.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029560.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031809.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030912.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030940.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030263.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029963.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030937.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029792.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032018.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032333.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 92%|█████████▏| 9235/10015 [01:05<00:06, 117.01it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032328.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030900.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030155.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032385.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 92%|█████████▏| 9261/10015 [01:05<00:06, 119.78it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029328.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030384.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029498.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031829.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031036.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029650.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029879.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029838.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030182.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 93%|█████████▎| 9285/10015 [01:05<00:07, 104.26it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030019.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030229.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030485.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031803.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032312.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030977.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031595.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031071.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031475.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031127.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029794.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 93%|█████████▎| 9309/10015 [01:06<00:06, 106.32it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029759.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030348.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032042.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031602.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030144.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029920.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031188.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030589.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030091.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 93%|█████████▎| 9333/10015 [01:06<00:06, 108.03it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029668.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030410.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031750.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030710.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030585.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030567.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030557.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032026.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030963.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030336.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031322.jpg"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 93%|█████████▎| 9357/10015 [01:06<00:05, 110.69it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031985.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030890.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030404.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031896.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032388.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031973.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029786.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032008.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031317.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029592.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 94%|█████████▎| 9381/10015 [01:06<00:05, 108.77it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030299.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029895.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030059.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032020.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029970.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030358.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029537.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031668.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030479.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030300.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 95%|█████████▌| 9542/10015 [01:07<00:01, 440.67it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030086.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031395.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034265.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033011.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032706.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033445.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032697.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032589.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032931.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033140.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033313.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033234.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033765.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033903.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032939.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033682.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033105.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032979.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033890.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033381.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033474.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034270.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032572.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032912.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029808.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031524.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034043.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029401.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033484.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033540.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034279.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033589.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033888.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033338.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032880.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032600.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032791.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032721.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033537.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032837.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032682.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033223.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032764.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032907.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033760.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033335.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034053.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032930.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032616.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033614.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033436.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033572.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033596.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034212.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033489.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033907.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033139.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033225.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033615.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033418.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034054.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033364.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034163.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034277.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034147.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033247.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033918.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032889.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033132.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033821.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033457.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033588.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033144.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033916.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032554.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033777.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033078.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034055.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033997.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032712.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033164.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033395.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033355.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033095.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033082.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034098.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033040.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034238.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032543.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033176.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032658.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032868.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032631.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032528.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034018.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033879.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033211.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033617.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033889.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033495.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033161.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032796.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033346.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034219.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033098.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032607.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033584.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033118.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032523.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033159.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033542.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032748.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034260.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034027.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032580.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033692.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032575.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032793.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033766.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033008.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033543.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034185.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033842.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032923.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032935.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034229.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032828.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033464.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033577.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033895.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032678.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033970.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033896.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034250.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033690.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034193.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033208.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032402.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034309.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032900.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033898.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033173.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032647.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 96%|█████████▌| 9634/10015 [01:07<00:00, 569.68it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032800.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033622.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032655.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033290.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034072.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034177.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033681.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033741.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033341.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032849.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033043.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032632.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034302.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032635.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034114.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032562.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033826.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033421.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033602.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034248.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033467.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033797.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033772.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033039.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032996.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032591.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034228.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033556.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033892.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032571.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033101.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033623.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034075.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032606.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033066.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033383.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033858.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034081.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033072.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032759.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033647.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032634.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032956.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033481.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034082.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033567.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034312.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033025.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033053.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033374.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033877.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032578.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033841.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033882.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034266.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033751.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033065.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033375.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033628.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033733.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032738.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033353.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034019.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032852.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034080.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034154.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032789.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033417.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033782.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033332.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034130.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032945.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033597.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034045.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033028.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033809.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033187.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032814.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033190.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033535.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033330.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033064.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033515.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033735.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033982.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034195.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033620.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0034040.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029355.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031259.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031216.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031769.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031209.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032379.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029368.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031105.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031359.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031454.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030038.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030909.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031323.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031791.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031488.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031148.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 97%|█████████▋| 9692/10015 [01:07<00:01, 277.66it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031222.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032102.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032273.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031299.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031547.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032221.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030693.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031649.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029417.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029915.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029360.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029659.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030586.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031040.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031929.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032173.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033456.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030408.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032329.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031198.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031191.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030242.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031119.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030297.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030714.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029634.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031918.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030803.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030602.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031874.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032437.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 97%|█████████▋| 9736/10015 [01:08<00:01, 203.71it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030076.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030191.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031043.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032455.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030794.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031738.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029930.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032349.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029362.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030991.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029541.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 98%|█████████▊| 9771/10015 [01:08<00:01, 171.30it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030827.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031940.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029582.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031044.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031852.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031609.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032422.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030826.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031012.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029533.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029563.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030491.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029715.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029860.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030341.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 98%|█████████▊| 9799/10015 [01:08<00:01, 159.82it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029315.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029811.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031743.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029932.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030375.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030175.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031335.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031692.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032203.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029371.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031659.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030707.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031211.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 98%|█████████▊| 9822/10015 [01:08<00:01, 157.92it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030953.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030821.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031292.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031570.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032277.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033494.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 98%|█████████▊| 9861/10015 [01:09<00:01, 143.78it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032371.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029610.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033413.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030785.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029840.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030143.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032154.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032356.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029567.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030387.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031381.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029851.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029573.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030280.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031578.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032397.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030245.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030487.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032238.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031672.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030986.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031431.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 99%|█████████▉| 9894/10015 [01:09<00:00, 128.88it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029598.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029781.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032897.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030730.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030549.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032014.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029462.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r 99%|█████████▉| 9908/10015 [01:09<00:00, 117.29it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031421.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030591.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029638.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031286.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030158.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029900.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029549.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 99%|█████████▉| 9934/10015 [01:09<00:00, 112.31it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033000.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033295.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031228.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030142.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031823.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029460.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030844.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031993.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029500.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032404.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031927.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            " 99%|█████████▉| 9958/10015 [01:10<00:00, 102.57it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030344.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032135.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030463.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029827.jpg"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|█████████▉| 9980/10015 [01:10<00:00, 102.48it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032206.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032199.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033869.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030825.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029713.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033866.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030036.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031506.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|█████████▉| 10007/10015 [01:10<00:00, 111.36it/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029830.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031108.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030133.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033811.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030877.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033358.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0030655.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033151.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031922.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032947.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0029309.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033705.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0031430.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033084.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033550.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0033536.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032854.jpg\n",
            "⚠️ Warning: Image not found or cannot be read -> /content/skin_cancer_data/ham10000_images_part_1/ISIC_0032258.jpg\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r100%|██████████| 10015/10015 [01:10<00:00, 141.93it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m157/157\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1009s\u001b[0m 6s/step\n",
            "Extracted Features Shape: (5000, 2048)\n",
            "Features saved as CSV: /content/drive/MyDrive/HAM10000_features.csv and NumPy: /content/drive/MyDrive/HAM10000_features.npy\n",
            "SVM model saved to: /content/drive/MyDrive/HAM10000_SVM_model.pkl\n",
            "SVM Accuracy: 79.80%\n",
            "Classification Report:\n",
            "               precision    recall  f1-score   support\n",
            "\n",
            "       akiec       0.61      0.30      0.40        37\n",
            "         bcc       0.54      0.51      0.52        53\n",
            "         bkl       0.66      0.53      0.59       113\n",
            "          df       0.00      0.00      0.00        11\n",
            "         mel       0.65      0.30      0.41        87\n",
            "          nv       0.84      0.98      0.90       686\n",
            "        vasc       1.00      0.23      0.38        13\n",
            "\n",
            "    accuracy                           0.80      1000\n",
            "   macro avg       0.61      0.41      0.46      1000\n",
            "weighted avg       0.77      0.80      0.77      1000\n",
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n",
            "/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n",
            "/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n"
          ]
        }
      ],
      "source": [
        "\n",
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import cv2\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.applications import ResNet50, ResNet101\n",
        "from tensorflow.keras.applications.resnet50 import preprocess_input\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "import joblib\n",
        "from tqdm import tqdm\n",
        "\n",
        "data_dir = \"/content/skin_cancer_data/ham10000_images_part_1\"\n",
        "metadata_path = \"/content/skin_cancer_data/HAM10000_metadata.csv\"\n",
        "\n",
        "df = pd.read_csv(metadata_path)\n",
        "print(df.head())\n",
        "\n",
        "IMG_SIZE = (224, 224)\n",
        "resnet_model = ResNet50(weights=\"imagenet\", include_top=False, pooling=\"avg\")\n",
        "\n",
        "def load_and_preprocess_image(image_path):\n",
        "    img = cv2.imread(image_path)\n",
        "\n",
        "    if img is None:\n",
        "        print(f\"Sorry Bro !!!!  Warning: Image not found or cannot be read -> {image_path}\")\n",
        "        return None\n",
        "\n",
        "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
        "    img = cv2.resize(img, IMG_SIZE)\n",
        "    img = preprocess_input(img)\n",
        "    return img\n",
        "\n",
        "images = []\n",
        "labels = []\n",
        "\n",
        "for i, row in tqdm(df.iterrows(), total=len(df)):\n",
        "    img_path = os.path.join(data_dir, row['image_id'] + \".jpg\")\n",
        "    img = load_and_preprocess_image(img_path)\n",
        "\n",
        "    if img is None:\n",
        "        continue\n",
        "\n",
        "    images.append(img)\n",
        "    labels.append(row['dx'])\n",
        "\n",
        "images = np.array(images)\n",
        "labels = np.array(labels)\n",
        "\n",
        "encoder = LabelEncoder()\n",
        "labels_encoded = encoder.fit_transform(labels)\n",
        "\n",
        "features = resnet_model.predict(images, batch_size=32, verbose=1)\n",
        "print(f\"Extracted Features Shape: {features.shape}\")\n",
        "\n",
        "csv_path = \"/content/drive/MyDrive/HAM10000_features.csv\"\n",
        "np_path = \"/content/drive/MyDrive/HAM10000_features.npy\"\n",
        "np.savetxt(csv_path, features, delimiter=\",\")\n",
        "np.save(np_path, features)\n",
        "print(f\"Features saved as CSV: {csv_path} and NumPy: {np_path}\")\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)\n",
        "\n",
        "svm_model = SVC(kernel=\"rbf\", C=1.0, gamma=\"scale\")\n",
        "svm_model.fit(X_train, y_train)\n",
        "\n",
        "svm_model_path = \"/content/drive/MyDrive/HAM10000_SVM_model.pkl\"\n",
        "joblib.dump(svm_model, svm_model_path)\n",
        "print(f\"SVM model saved to: {svm_model_path}\")\n",
        "\n",
        "y_pred = svm_model.predict(X_test)\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(f\"SVM Accuracy: {accuracy * 100:.2f}%\")\n",
        "\n",
        "print(\"Classification Report:\\n\", classification_report(y_test, y_pred, target_names=encoder.classes_))\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(f\"Total images successfully loaded: {len(images)}\")\n",
        "print(f\" Total missing/corrupted images: {len(df) - len(images)}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Esi9ZJVRF1Ps",
        "outputId": "20f72581-3e75-4fd7-a231-52f06db003ed"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Total images successfully loaded: 5000\n",
            "⚠️ Total missing/corrupted images: 5015\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.svm import SVC\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "import joblib\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)\n",
        "\n",
        "svm_model = SVC(kernel=\"rbf\", C=1.0, gamma=\"scale\")\n",
        "svm_model.fit(X_train, y_train)\n",
        "\n",
        "svm_model_path = \"/content/drive/MyDrive/HAM10000_SVM_model.pkl\"\n",
        "joblib.dump(svm_model, svm_model_path)\n",
        "print(f\"SVM Model Saved: {svm_model_path}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cxECo2TcGlzL",
        "outputId": "135aca03-55fc-4eba-9c87-f93ee94f17bb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ SVM Model Saved: /content/drive/MyDrive/HAM10000_SVM_model.pkl\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "y_pred = svm_model.predict(X_test)\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(f\" SVM Accuracy: {accuracy * 100:.2f}%\")\n",
        "print(\"Classification Report:\\n\", classification_report(y_test, y_pred, target_names=encoder.classes_))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "o9k0-qYdHeAF",
        "outputId": "8cca3f7e-dcdf-45d1-df00-930bab132150"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ SVM Accuracy: 79.80%\n",
            "🔍 Classification Report:\n",
            "               precision    recall  f1-score   support\n",
            "\n",
            "       akiec       0.61      0.30      0.40        37\n",
            "         bcc       0.54      0.51      0.52        53\n",
            "         bkl       0.66      0.53      0.59       113\n",
            "          df       0.00      0.00      0.00        11\n",
            "         mel       0.65      0.30      0.41        87\n",
            "          nv       0.84      0.98      0.90       686\n",
            "        vasc       1.00      0.23      0.38        13\n",
            "\n",
            "    accuracy                           0.80      1000\n",
            "   macro avg       0.61      0.41      0.46      1000\n",
            "weighted avg       0.77      0.80      0.77      1000\n",
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n",
            "/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n",
            "/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
            "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", len(result))\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras import models, layers\n",
        "\n",
        "for layer in resnet_model.layers[:-1]:\n",
        "    layer.trainable = False\n",
        "\n",
        "model = models.Sequential()\n",
        "model.add(resnet_model)\n",
        "model.add(layers.Dense(256, activation='relu'))\n",
        "model.add(layers.Dense(len(np.unique(labels_encoded)), activation='softmax'))\n",
        "\n",
        "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "history = model.fit(images, labels_encoded, epochs=5, batch_size=32, validation_split=0.2)\n",
        "\n",
        "loss, accuracy = model.evaluate(images, labels_encoded)\n",
        "print(f\"Fine-Tuned ResNet Accuracy: {accuracy * 100:.2f}%\")\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 391
        },
        "id": "rlcUBRVyIOcr",
        "outputId": "8967cb87-5829-400b-cfa5-c87573006925"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "\u001b[1m125/125\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6s/step - accuracy: 0.7121 - loss: 1.0275"
          ]
        },
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-9-81c6bf843ae8>\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcompile\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moptimizer\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'adam'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mloss\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'sparse_categorical_crossentropy'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmetrics\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'accuracy'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     12\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 13\u001b[0;31m \u001b[0mhistory\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimages\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabels_encoded\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mepochs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m5\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m32\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalidation_split\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0.2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     14\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0mloss\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maccuracy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mevaluate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mimages\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlabels_encoded\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py\u001b[0m in \u001b[0;36merror_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    115\u001b[0m         \u001b[0mfiltered_tb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    116\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 117\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    118\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    119\u001b[0m             \u001b[0mfiltered_tb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_process_traceback_frames\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__traceback__\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq)\u001b[0m\n\u001b[1;32m    393\u001b[0m                         \u001b[0mshuffle\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    394\u001b[0m                     )\n\u001b[0;32m--> 395\u001b[0;31m                 val_logs = self.evaluate(\n\u001b[0m\u001b[1;32m    396\u001b[0m                     \u001b[0mx\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mval_x\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    397\u001b[0m                     \u001b[0my\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mval_y\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py\u001b[0m in \u001b[0;36merror_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    115\u001b[0m         \u001b[0mfiltered_tb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    116\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 117\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    118\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    119\u001b[0m             \u001b[0mfiltered_tb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_process_traceback_frames\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__traceback__\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py\u001b[0m in \u001b[0;36mevaluate\u001b[0;34m(self, x, y, batch_size, verbose, sample_weight, steps, callbacks, return_dict, **kwargs)\u001b[0m\n\u001b[1;32m    482\u001b[0m             \u001b[0;32mfor\u001b[0m \u001b[0mstep\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0miterator\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mepoch_iterator\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    483\u001b[0m                 \u001b[0mcallbacks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_test_batch_begin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 484\u001b[0;31m                 \u001b[0mlogs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtest_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    485\u001b[0m                 \u001b[0mcallbacks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_test_batch_end\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlogs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    486\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstop_evaluating\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py\u001b[0m in \u001b[0;36mfunction\u001b[0;34m(iterator)\u001b[0m\n\u001b[1;32m    217\u001b[0m                 \u001b[0miterator\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mIterator\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdistribute\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mDistributedIterator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    218\u001b[0m             ):\n\u001b[0;32m--> 219\u001b[0;31m                 \u001b[0mopt_outputs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmulti_step_on_iterator\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    220\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mopt_outputs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhas_value\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    221\u001b[0m                     \u001b[0;32mraise\u001b[0m \u001b[0mStopIteration\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py\u001b[0m in \u001b[0;36merror_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    148\u001b[0m     \u001b[0mfiltered_tb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    149\u001b[0m     \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 150\u001b[0;31m       \u001b[0;32mreturn\u001b[0m \u001b[0mfn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    151\u001b[0m     \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    152\u001b[0m       \u001b[0mfiltered_tb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_process_traceback_frames\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__traceback__\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    831\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    832\u001b[0m       \u001b[0;32mwith\u001b[0m \u001b[0mOptionalXlaContext\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_jit_compile\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 833\u001b[0;31m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    834\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    835\u001b[0m       \u001b[0mnew_tracing_count\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexperimental_get_tracing_count\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py\u001b[0m in \u001b[0;36m_call\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    876\u001b[0m       \u001b[0;31m# In this case we have not created variables on the first call. So we can\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    877\u001b[0m       \u001b[0;31m# run the first trace but we should fail if variables are created.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 878\u001b[0;31m       results = tracing_compilation.call_function(\n\u001b[0m\u001b[1;32m    879\u001b[0m           \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_variable_creation_config\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    880\u001b[0m       )\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py\u001b[0m in \u001b[0;36mcall_function\u001b[0;34m(args, kwargs, tracing_options)\u001b[0m\n\u001b[1;32m    137\u001b[0m   \u001b[0mbound_args\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfunction\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfunction_type\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbind\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    138\u001b[0m   \u001b[0mflat_inputs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfunction\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfunction_type\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munpack_inputs\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbound_args\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 139\u001b[0;31m   return function._call_flat(  # pylint: disable=protected-access\n\u001b[0m\u001b[1;32m    140\u001b[0m       \u001b[0mflat_inputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcaptured_inputs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mfunction\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcaptured_inputs\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    141\u001b[0m   )\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py\u001b[0m in \u001b[0;36m_call_flat\u001b[0;34m(self, tensor_inputs, captured_inputs)\u001b[0m\n\u001b[1;32m   1320\u001b[0m         and executing_eagerly):\n\u001b[1;32m   1321\u001b[0m       \u001b[0;31m# No tape is watching; skip to running the function.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1322\u001b[0;31m       \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_inference_function\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcall_preflattened\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1323\u001b[0m     forward_backward = self._select_forward_and_backward_functions(\n\u001b[1;32m   1324\u001b[0m         \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py\u001b[0m in \u001b[0;36mcall_preflattened\u001b[0;34m(self, args)\u001b[0m\n\u001b[1;32m    214\u001b[0m   \u001b[0;32mdef\u001b[0m \u001b[0mcall_preflattened\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mSequence\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mTensor\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0mAny\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    215\u001b[0m     \u001b[0;34m\"\"\"Calls with flattened tensor inputs and returns the structured output.\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 216\u001b[0;31m     \u001b[0mflat_outputs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcall_flat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    217\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfunction_type\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpack_output\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mflat_outputs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    218\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py\u001b[0m in \u001b[0;36mcall_flat\u001b[0;34m(self, *args)\u001b[0m\n\u001b[1;32m    249\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mrecord\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstop_recording\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    250\u001b[0m           \u001b[0;32mif\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_bound_context\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mexecuting_eagerly\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 251\u001b[0;31m             outputs = self._bound_context.call_function(\n\u001b[0m\u001b[1;32m    252\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    253\u001b[0m                 \u001b[0mlist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/context.py\u001b[0m in \u001b[0;36mcall_function\u001b[0;34m(self, name, tensor_inputs, num_outputs)\u001b[0m\n\u001b[1;32m   1681\u001b[0m     \u001b[0mcancellation_context\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcancellation\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcontext\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1682\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mcancellation_context\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1683\u001b[0;31m       outputs = execute.execute(\n\u001b[0m\u001b[1;32m   1684\u001b[0m           \u001b[0mname\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdecode\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"utf-8\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1685\u001b[0m           \u001b[0mnum_outputs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnum_outputs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py\u001b[0m in \u001b[0;36mquick_execute\u001b[0;34m(op_name, num_outputs, inputs, attrs, ctx, name)\u001b[0m\n\u001b[1;32m     51\u001b[0m   \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     52\u001b[0m     \u001b[0mctx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mensure_initialized\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 53\u001b[0;31m     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,\n\u001b[0m\u001b[1;32m     54\u001b[0m                                         inputs, attrs, num_outputs)\n\u001b[1;32m     55\u001b[0m   \u001b[0;32mexcept\u001b[0m \u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_NotOkStatusException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "gKhwNqkNbxRs"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}