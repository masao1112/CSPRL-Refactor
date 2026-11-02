# CSPRL-Refactor

### Trang giới thiệu công trình nghiên cứu

Trang nghiên cứu với ảnh động minh họa quy trình, các bản đồ mới trong `res/`, kiến trúc và số liệu từ bài báo camera-ready. Không nạp mô hình; luồng giả lập được dựng sẵn thành GIF.

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory demo
```

Mở <http://localhost:8765>. Xem [hướng dẫn trang nghiên cứu](demo/README.md) để biết nguồn ảnh, số liệu bài báo và cách dựng GIF minh họa.

Tối ưu vị trí và cấu hình trạm sạc xe điện trên mạng đường bộ Hà Nội bằng học tăng cường. Môi trường Gymnasium kết hợp nhu cầu sạc, chi phí di chuyển/chờ/sạc, độ công bằng và khả năng đáp ứng của lưới điện. Agent lựa chọn các chiến lược xây mới, mở rộng hoặc di chuyển bộ sạc; các heuristic trong môi trường quyết định vị trí cụ thể.

Repo hỗ trợ **DQN**, **PPO**, policy **MLP / MLP với đặc trưng đồ thị / GNN / attention**, cùng các baseline **greedy** và **genetic algorithm (GA)**. Dữ liệu đồ thị, kế hoạch trạm hiện có và lưới điện đã được lưu trong repo; chưa có checkpoint huấn luyện đi kèm.

## Cấu trúc

```text
.
├── train.py                       # Huấn luyện DQN/PPO, lưu cấu hình và checkpoint
├── evaluate_all.py                # Đánh giá và xếp hạng checkpoint của một lần chạy
├── run_configs.py                 # Chạy tuần tự các thí nghiệm trong CONFIGS
├── run_metrics.py                 # Tính chỉ số cho kế hoạch đã xuất
├── preprocess_scaling.py          # Tạo thống kê chuẩn hóa theo quận
├── algorithms/
│   ├── compare_rl.py              # So sánh RL, GA, greedy và hạ tầng hiện tại
│   ├── greedy_benefit.py
│   ├── greedy_demand.py
│   └── ga/                        # Tiến hóa trọng số policy MLP
├── custom_environment/
│   ├── StationPlacementEnv.py     # Môi trường, trạng thái, hành động và reward
│   ├── helpers.py                 # Mô hình nhu cầu, chi phí, ràng buộc và hằng số
│   ├── graph_features.py          # Đặc trưng lân cận cho mlp_graph
│   ├── gnn_extractor.py           # GCN viết bằng PyTorch
│   ├── attention_extractor.py     # Self-attention và pooling
│   ├── power_grid/                # Đọc/sinh lưới điện và tính grid penalty
│   └── data/                     # Graph/, hanoi_citywide/, processed/, QGIS/
└── src/
    ├── preprocessing/             # Sinh bảng cấu hình bộ sạc và kế hoạch hiện tại
    └── analysis/plot_plan_maps.py  # Xuất bản đồ kế hoạch và mã LaTeX
```

`evaluate.py`, `visualise.py` và `export_config.py` là các tiện ích cũ có location, đường dẫn hoặc bước checkpoint gắn cứng trong mã. Luồng CLI bên dưới dùng `evaluate_all.py` và `algorithms/compare_rl.py`.

## Cài đặt

Chạy các lệnh từ thư mục gốc của repo. Có thể tạo môi trường Python 3.12 riêng:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python train.py --help
```

Trên Windows, kích hoạt bằng `.venv\Scripts\activate`.

Các thư viện chính gồm PyTorch, Stable-Baselines3, Gymnasium, OSMnx, NetworkX, GeoPandas và pandapower. GNN hiện dùng trực tiếp PyTorch, không cần PyTorch Geometric. Có thể chọn CPU bằng `--device cpu`; CUDA cần bản PyTorch và driver tương thích.

`requirements.txt` quy định phiên bản tối thiểu, chưa phải lockfile. `train.py` import `LinearSchedule` từ Stable-Baselines3: môi trường dùng bản SB3 cũ không có lớp này sẽ lỗi ngay khi import; cần cập nhật SB3 trong môi trường đó. Sau khi có bộ dependencies chạy ổn, lưu lại phiên bản bằng `python -m pip freeze > requirements-local.txt` để tái lập thí nghiệm.

## Dữ liệu đầu vào

Mỗi quận dùng các file trong `custom_environment/data/Graph/<location>/`:

| File | Vai trò |
| --- | --- |
| `<location>.graphml` | Mạng đường bộ |
| `nodes_extended_<location>.txt` | Danh sách node và thuộc tính nhu cầu, dân số, giá đất, tọa độ |
| `new_existingplan_<location>.pkl` | Kế hoạch đầu vào của `train.py`, `evaluate_all.py` và GA |
| `existingplan_<location>.pkl` | Kế hoạch dùng bởi các script greedy cũ; một số công cụ có fallback về file này |
| `scaling_constants.json` | Miền chuẩn hóa; môi trường dùng giá trị mặc định nếu thiếu |

Dữ liệu hiện có:

| Location | Graph + node | Existing plan | New existing plan + scaling |
| --- | --- | --- | --- |
| `DongDa`, `CauGiay`, `NamTuLiem`, `TayHo` | Có | Có | Có |
| `BaDinh`, `HaiBaTrung`, `ThanhXuan` | Có | Có | Chưa có |
| `HoanKiem`, `OCD` | Có | Chưa có | Chưa có |

Bắt đầu với `DongDa` để dùng đầy đủ luồng huấn luyện và đánh giá. Không tự coi hai loại kế hoạch `existingplan` và `new_existingplan` là tương đương khi so sánh kết quả.

Adapter ưu tiên lưới `data/hanoi_<location viết thường>/`, sau đó dùng `data/hanoi_citywide/`. Bộ CSV lưới toàn thành phố đã có sẵn. Nếu adapter khởi tạo thất bại, môi trường có thể tiếp tục với fallback; kiểm tra log để biết lần chạy có thực sự dùng lưới điện hay không.

Node TXT được một số script đọc bằng `eval`, còn kế hoạch dùng pickle: chỉ sử dụng dữ liệu từ nguồn tin cậy.

## Huấn luyện

Ví dụ DQN với MLP, seed và namespace riêng:

```bash
python train.py --algo dqn --location DongDa --obs_type mlp \
  --total_timesteps 200000 --seed 1 --ns mlp_s1 --device cpu
```

Có thể đổi `--obs_type`:

| Giá trị | Biểu diễn |
| --- | --- |
| `mlp` | Vector đặc trưng node và trạng thái toàn cục |
| `mlp_graph` | Vector bổ sung thông tin lân cận một bước và thống kê đồ thị |
| `gnn` | Dict observation với node, cạnh và trạng thái toàn cục; trích xuất bằng GCN |
| `attention` | Dict observation như GNN; attention xử lý tập node và bỏ qua cạnh |

Mặc định khi không truyền `--obs_type` là `gnn`. `--no_gnn` chọn MLP theo giao diện cũ. Với GNN hiện tại, `--features_dim` được giữ để tương thích nhưng kích thước đầu ra thực tế là `số node + 32`.

Ví dụ PPO với attention:

```bash
python train.py --algo ppo --location DongDa --obs_type attention \
  --embed_dim 128 --n_heads 4 --n_layers 2 \
  --n_steps 2048 --batch_size 128 --total_timesteps 200000 \
  --seed 1 --ns attention_s1 --device cpu
```

Attention yêu cầu `embed_dim` chia hết cho `n_heads`. `--no_pma` chuyển pooling sang mean-pooling. Self-attention và ma trận kề đặc của GNN có thể cần nhiều bộ nhớ trên đồ thị lớn.

Các tùy chọn khác xem bằng `python train.py --help`. Một số giá trị mặc định:

- Learning rate `1e-4`, batch size `128`, seed `1`, tổng bước `200000`.
- DQN: replay buffer `20000`, epsilon từ `1.0` đến `0.05`.
- Ablation: `--grid_penalty_weight 1.0`, `--eta 0.4`, `--beta 0.6`.
- `--lr_schedule` bật giảm learning rate tuyến tính.

Hằng số đang được thực thi nằm trong `custom_environment/helpers.py`, gồm ngân sách `900000`, tối đa `100` bộ sạc/trạm và các loại công suất. `custom_environment/env_configs.json` hiện chưa được mã nguồn nạp và chứa một số giá trị khác; chỉnh file JSON này chưa làm thay đổi mô phỏng.

Để chạy nhiều cấu hình, chỉnh `CONFIGS` trong `run_configs.py`, rồi chạy:

```bash
python run_configs.py --only 0
python run_configs.py --start_from 3
```

Mỗi cấu hình chạy trong subprocess riêng và được xử lý tuần tự.

## Đánh giá và so sánh

Sau khi huấn luyện, đánh giá các checkpoint trong đúng thư mục của lần chạy:

```bash
python evaluate_all.py --path_dir Results/tmp/DongDa/mlp/mlp_s1 \
  --min_step 0 --episodes 3 --metric score --device cpu
```

Mặc định `--min_step` là `60000`; đặt `0` để đánh giá cả checkpoint sớm. Công cụ đọc `config.json` để xác định observation, location và các tham số reward, sau đó lưu `evaluation_results.csv` và `evaluation_plot.png` ngay trong thư mục checkpoint. Giữ nguyên tên/thư mục PPO do `train.py` tạo vì evaluator nhận diện PPO qua chuỗi `ppo` trong đường dẫn.

Huấn luyện GA nếu muốn có kết quả baseline GA:

```bash
python algorithms/ga/train_ga.py --location DongDa --seed 1 --ns ga_s1 \
  --pop_size 50 --generations 100
```

So sánh với DQN đã huấn luyện:

```bash
MPLBACKEND=Agg python algorithms/compare_rl.py --location DongDa \
  --algo dqn --obs_type mlp --ns mlp_s1 --ga_ns ga_s1 --device cpu
```

`compare_rl.py` chọn checkpoint theo `evaluation_results.csv` nếu có; có thể chỉ định `--step <bước>`. Với PPO, dùng `--algo ppo`, observation và namespace tương ứng. `--algo both --ppo_ns <namespace>` so sánh cả DQN và PPO. GA chỉ được nạp khi tìm thấy model; đọc log để biết các agent thực sự được đánh giá.

Công cụ so sánh xuất chỉ số, biểu đồ, bản đồ và các cặp `plan_*.pkl` / `nodes_*.txt`. Có thể tính lại chỉ số của một kế hoạch bằng:

```bash
python run_metrics.py --location DongDa \
  --plan_file Results/optimal_plan/DongDa/plan_DQN_<step>.pkl \
  --node_file Results/optimal_plan/DongDa/nodes_DQN_<step>.txt
```

Thay `<step>` bằng bước trong tên file được xuất. Khi so sánh, dùng cùng kế hoạch đầu vào, seed, số episode, tham số nhu cầu và trọng số lưới. `--max_steps` của công cụ so sánh chỉ dành cho chạy thử ngắn; điểm của episode bị cắt không tương đương episode đầy đủ.

## Kết quả và mô hình điểm

```text
Results/
├── tmp/<location>/<obs_type>/<ns>/       # DQN
├── tmp/<location>/<obs_type>/ppo/<ns>/   # PPO
├── ga/<location>/<ns>/                  # GA
├── maps/<location>/                     # Bản đồ so sánh
└── optimal_plan/<location>/             # Kế hoạch và node đã xuất
```

Nếu bỏ `--ns`, kết quả nằm trực tiếp trong thư mục cha tương ứng. Dùng namespace riêng cho mỗi cấu hình/seed để tránh ghi đè.

Mỗi lần huấn luyện RL ghi `config.json` trước khi chạy, log `monitor.csv` và checkpoint `best_model_*.zip` khi đạt tiêu chí lưu. `episode_history.csv` và `training_plot.png` được tạo sau khi huấn luyện hoàn tất. Lần chạy quá ngắn có thể chưa kết thúc episode nên chưa có checkpoint.

Môi trường có 5 hành động:

| ID | Chiến lược |
| --- | --- |
| `0` | Xây trạm mới theo benefit |
| `1` | Xây trạm mới theo demand |
| `2` | Bổ sung bộ sạc theo benefit |
| `3` | Bổ sung bộ sạc theo demand |
| `4` | Di chuyển bộ sạc từ trạm có thể điều chỉnh sang vị trí cần hỗ trợ |

Score kết hợp benefit, chi phí và fairness được chuẩn hóa, rồi trừ grid penalty theo khoảng cách và quá tải. Reward mỗi bước là thay đổi score cộng bonus khi vượt score tốt nhất của episode. Các chỉ số chi tiết được tính trong `helpers.py` và grid adapter; tổng reward và score là hai đại lượng khác nhau.

## Tiền xử lý và trực quan hóa

Dữ liệu cần cho ví dụ `DongDa` đã có sẵn. Chỉ tái tạo khi chủ động thay dữ liệu hoặc tham số:

```bash
# Ghi lại scaling_constants.json của quận
python preprocess_scaling.py DongDa

# Ghi lại bảng cấu hình bộ sạc bằng quy hoạch động; có thể tốn thời gian/bộ nhớ
python src/preprocessing/create_lookup.py

# Xuất bản đồ từ một kế hoạch có sẵn
python src/analysis/plot_plan_maps.py --location DongDa \
  "Existing infrastructure=custom_environment/data/Graph/DongDa/new_existingplan_DongDa.pkl"
```

`src/preprocessing/prepare_exisitingplan.py` ánh xạ trạm từ CSV QGIS sang node và ghi `existingplan_<location>.pkl`; location hiện gắn cứng trong script. Công cụ này không tạo `new_existingplan_*.pkl`. `search_params.py` cũng dùng location gắn cứng và thực hiện quét tham số nhu cầu.

`custom_environment/power_grid/citywide_generator.py` có thể sinh lại CSV lưới điện. Đây là dữ liệu mô phỏng; các đầu vào đường bộ/POI/dân số phục vụ tái tạo đầy đủ không phải đều có trong repo. `debug_grid_pf.py` còn tham chiếu package/path `CSPRL` cũ, cần điều chỉnh trước khi dùng.

## Kiểm tra mã

Repo hiện chưa có bộ kiểm thử tự động. Có thể kiểm tra cú pháp bằng:

```bash
python -m compileall -q train.py evaluate_all.py algorithms custom_environment src
```

Với môi trường đã cài dependencies, dùng `--help` để kiểm tra CLI và chạy một thí nghiệm nhỏ trước khi huấn luyện dài. Không dùng kết quả chạy thử ngắn làm số liệu so sánh chính thức. Các kết quả thí nghiệm, cache Python và cấu hình IDE được bỏ qua trong `.gitignore`.
