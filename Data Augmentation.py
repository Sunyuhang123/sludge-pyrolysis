import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Input, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# 设置字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，适合中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建条件GAN模型
class cGAN:
    def __init__(self, input_dim, output_dim, label_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.label_dim = label_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        noise_input = Input(shape=(self.input_dim,))
        label_input = Input(shape=(self.label_dim,))
        combined_input = Concatenate()([noise_input, label_input])
        x = Dense(128)(combined_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(self.output_dim, activation='relu')(x)  # 使用ReLU确保输出非负
        model = Model([noise_input, label_input], x)
        return model

    def build_discriminator(self):
        data_input = Input(shape=(self.output_dim,))
        label_input = Input(shape=(self.label_dim,))
        combined_input = Concatenate()([data_input, label_input])
        x = Dense(256)(combined_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(128)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model([data_input, label_input], x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        noise_input = Input(shape=(self.input_dim,))
        label_input = Input(shape=(self.label_dim,))
        generated_data = self.generator([noise_input, label_input])
        validity = self.discriminator([generated_data, label_input])
        model = Model([noise_input, label_input], validity)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, X_train, y_train, epochs, batch_size):
        half_batch = batch_size // 2
        for epoch in range(epochs):
            # 随机选择一半批次的真实数据
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            real_data = X_train[idx]
            real_labels = y_train[idx]

            # 生成一半批次的假数据
            noise = np.random.normal(0, 1, (half_batch, self.input_dim))
            fake_data = self.generator.predict([noise, real_labels])

            # 训练判别器
            d_loss_real = self.discriminator.train_on_batch([real_data, real_labels], np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch([fake_data, real_labels], np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            valid_y = np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch([noise, y_train[np.random.randint(0, y_train.shape[0], batch_size)]], valid_y)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

# 加载数据
data = pd.read_excel(r"D:\北林材料\11.25号数据.xlsx", engine='openpyxl')
SYHFeatures = data.iloc[:, :-3]
SYHLabels = data.iloc[:, -3:]

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(SYHFeatures.values)

# 创建桌面保存结果的文件夹路径
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop', 'FigureSave50数据')
if not os.path.exists(desktop_path):
    os.makedirs(desktop_path)

# 存储所有评价指标
all_metrics = []

# GAN数据增强并训练
num_samples_list = [200, 500, 1000, 2000, 5000]
for i in range(SYHLabels.shape[1]):  # 遍历每个目标变量
    print(f"Processing target variable {i}")

    for num_samples in num_samples_list:
        print(f"Generating {num_samples} samples using GAN for target {i}")

        # 创建条件GAN模型
        gan = cGAN(input_dim=SYHFeatures.shape[1], output_dim=SYHFeatures.shape[1], label_dim=SYHLabels.shape[1])
        gan.train(X_scaled, SYHLabels.values, epochs=1000, batch_size=64)

        # 生成数据
        noise = np.random.normal(0, 1, (num_samples, SYHFeatures.shape[1]))
        conditions = SYHLabels.values[np.random.choice(SYHLabels.shape[0], num_samples)]  # 随机选择真实标签作为条件
        generated_data = gan.generator.predict([noise, conditions])

        # 裁剪生成的数据，确保不小于0
        generated_data = np.maximum(generated_data, 0)

        # 合并数据
        augmented_data = np.vstack([X_scaled, generated_data])
        augmented_labels = np.vstack([SYHLabels.values, conditions])

        # 确保数据充分洗牌
        augmented_data, augmented_labels = shuffle(augmented_data, augmented_labels, random_state=42)

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(augmented_data, augmented_labels[:, i], test_size=0.2, random_state=42)

        # 训练 XGBoost 模型
        rf_reg = xgb.XGBRegressor(learning_rate=0.1, max_depth=5, n_estimators=100, objective='reg:squarederror')
        rf_reg.fit(X_train, y_train)

        # 预测
        y_train_pred = rf_reg.predict(X_train)
        y_test_pred = rf_reg.predict(X_test)

        # 计算评价指标
        val_r2 = r2_score(y_true=y_test, y_pred=y_test_pred)
        val_mse = mean_squared_error(y_true=y_test, y_pred=y_test_pred) ** 0.5  # RMSE
        val_mae = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)

        # 打印测试集评价结果
        print(f"Test R2: {val_r2:.2f}")
        print(f"Test MSE: {val_mse:.2f}")
        print(f"Test MAE: {val_mae:.2f}")

        # 存储评价指标
        metrics = {
            'Target': i,
            'Samples': num_samples,
            'R2': val_r2,
            'MSE': val_mse,
            'MAE': val_mae,
            'RMSE': val_mse
        }
        all_metrics.append(metrics)

        # 保存训练集和测试集的真实值与预测值到 Excel
        train_results = pd.DataFrame({
            'Actual': y_train,
            'Predicted': y_train_pred
        })
        test_results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_test_pred
        })

        # 保存训练集和测试集的结果到 Excel
        results_folder = os.path.join(desktop_path, f"Target_{i}_Samples_{num_samples}")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        train_results_file_path = os.path.join(results_folder, f"train_results_{num_samples}.xlsx")
        test_results_file_path = os.path.join(results_folder, f"test_results_{num_samples}.xlsx")

        train_results.to_excel(train_results_file_path, index=False)
        test_results.to_excel(test_results_file_path, index=False)

        print(f"Train and Test results saved for target {i}, samples {num_samples}")

        # 在 SHAP 值计算部分修改如下：

        # SHAP值计算
        explainer = shap.Explainer(rf_reg)
        X_train_df = pd.DataFrame(X_train, columns=SYHFeatures.columns)
        X_test_df = pd.DataFrame(X_test, columns=SYHFeatures.columns)
        shap_values_train = explainer(X_train_df)
        shap_values_test = explainer(X_test_df)

        # 计算 SHAP 绝对平均值
        shap_abs_values_train = np.abs(shap_values_train.values).mean(axis=0)  # 训练集 SHAP 绝对平均值
        shap_abs_values_test = np.abs(shap_values_test.values).mean(axis=0)  # 测试集 SHAP 绝对平均值

        # 创建 DataFrame 保存 SHAP 绝对平均值
        shap_abs_df = pd.DataFrame({
            'Feature': SYHFeatures.columns,  # 特征名称
            'SHAP Abs Mean (Train)': shap_abs_values_train,  # 训练集 SHAP 绝对平均值
            'SHAP Abs Mean (Test)': shap_abs_values_test  # 测试集 SHAP 绝对平均值
        })

        # 按训练集的 SHAP 绝对平均值排序
        shap_abs_df = shap_abs_df.sort_values(by='SHAP Abs Mean (Train)', ascending=False)

        # 保存 SHAP 绝对平均值到 Excel
        shap_abs_file_path = os.path.join(results_folder, f'SHAP_abs_mean_{num_samples}.xlsx')
        shap_abs_df.to_excel(shap_abs_file_path, index=False)
        print(f"SHAP absolute mean values saved to {shap_abs_file_path}")

        # 绘制SHAP蜂群图
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values_train, show=False)
        plt.title(f'SHAP Bee Swarm Plot for Target {i} with {num_samples} Samples')
        plt.savefig(os.path.join(results_folder, f'SHAP_beeswarm_{num_samples}.svg'), format='svg', dpi=600)
        plt.close()

# 保存所有评价指标到Excel
metrics_df = pd.DataFrame(all_metrics)
metrics_file_path = os.path.join(desktop_path, 'Evaluation_Metrics.xlsx')
metrics_df.to_excel(metrics_file_path, index=False)
print(f"Evaluation metrics saved to {metrics_file_path}")