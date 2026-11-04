# CSPRL — Trang nghiên cứu với demo quy hoạch bổ sung

Trang chính giữ nguyên format nghiên cứu: tiêu đề/tác giả, đầu ra, tóm tắt, bảng kết quả, diễn giải chỉ số, hình thu gọn, GIF và thông số mô hình. Tính năng quy hoạch được bổ sung trong mục có thể mở “Khám phá kế hoạch trên bản đồ Đống Đa”, đặt ở phần Luồng chạy. Mục này cho phép chọn phương pháp, tua kế hoạch, xem chi tiết trạm/thanh cái, so sánh hai phương án và xuất CSV.

## Chạy

Từ thư mục gốc repo:

~~~bash
python -m http.server 8765 --bind 127.0.0.1 --directory demo
~~~

Mở http://localhost:8765. Dùng trình duyệt hỗ trợ JavaScript modules. Trang cần HTTP server để đọc JSON; không dùng file://. Không cần npm, máy chủ mô hình hoặc internet. Các link đến GitHub chỉ được truy cập khi người dùng chủ động mở.

Mặc định mục demo đóng để giữ luồng đọc và bố cục cũ. Dữ liệu quy hoạch chỉ tải khi mở mục demo; đóng mục sẽ dừng replay. Nội dung nghiên cứu vẫn đọc được khi tắt JavaScript. CSS và ID của phần bổ sung được đặt trong phạm vi riêng, không thay đổi bảng số liệu hoặc phần trình bày bên ngoài.

## Luồng sử dụng

Trong phần **Luồng chạy**, mở **Khám phá kế hoạch trên bản đồ Đống Đa**:

1. Khi mở trang, hiển thị 5 trạm hiện hữu từ kế hoạch đầu vào và lớp nhu cầu nút giao.
2. Chọn DQN đề xuất, GA, Greedy-Benefit hoặc Greedy-Demand.
3. Bấm Chạy / Tạm dừng; dùng Tiến 1 bước, tốc độ hoặc timeline để xem lại. Đặt lại đưa phương pháp đang chọn về trạng thái 5 trạm đầu vào.
4. Bấm Kế hoạch tốt nhất để đến snapshot có R(p) cao nhất trong toàn bộ chuỗi của phương pháp đó. Nút này chọn theo điểm, không mặc định chọn bước cuối.
5. Chọn điểm trạm hoặc chọn trong danh sách để xem tọa độ, cấu hình 12 mức sạc, công suất, bán kính, chi phí, thời gian chờ và bus đấu nối. Bấm thanh cái để xem tải và dư địa.
6. Hai bản đồ so sánh hiển thị snapshot tốt nhất của hai phương pháp được chọn, dùng cùng tọa độ/tỉ lệ bản đồ và thang màu nhu cầu. Việc đổi phương án so sánh không thay đổi bước trên bản đồ chính.
7. Xuất CSV trên bản đồ chính lấy đúng snapshot đang xem, kể cả khi tua lại. Nút CSV trong mỗi phương án so sánh lấy đúng kế hoạch tốt nhất đang hiển thị ở phương án đó.

Các lớp Nhu cầu / Vùng phục vụ / Lưới điện áp dụng đồng thời cho cả ba bản đồ. Nhu cầu sử dụng cùng mốc chuẩn hóa ban đầu cho mọi phương án. Trạm hiện hữu màu xám, trạm mới màu đỏ; kích thước điểm tăng theo căn bậc hai công suất để diện tích điểm thể hiện quy mô. Vòng xanh đánh dấu lựa chọn và hành động; ô thoi là thanh cái. Tải sạc vượt 100% dư địa làm thanh cái chuyển đỏ.

## Dữ liệu và giới hạn mô phỏng

- Đường bộ: assets/roads.json xuất từ Graph/DongDa/DongDa.graphml, 670 nút hình học.
- Nút ứng viên và nhu cầu: 639 bản ghi trong nodes_extended_DongDa.txt; giá đất và tọa độ lấy trực tiếp từ repo.
- 5 trạm hiện hữu: new_existingplan_DongDa.pkl, đọc bằng unpickler chỉ chấp nhận kiểu dữ liệu nguyên thủy. Không dùng existingplan_DongDa.pkl (file đó có 17 trạm).
- Lưới: 20 thanh cái 22 kV gần nhất với các nút Đống Đa trong bộ CSV hanoi_citywide hiện tại, khác snapshot 18 thanh cái của bài báo. Không gán chỉ số từ bài báo cho các kế hoạch của app.
- Dư địa MW: min(định mức đường dây, định mức máy biến áp) × 0,8 − phụ tải nền CSV. Các bus thiếu định mức dùng fallback 10 MVA được đánh dấu trong dữ liệu và phần chi tiết. Không chạy power-flow.
- Mức sạc: 3, 7, 11, 20, 22, 30, 60, 80, 120, 150, 180, 250 kW, cùng INSTALL_FEE và BUDGET từ helpers.py.
- Chi phí và ngân sách dùng **triệu đồng (million_vnd)**; giá đất là triệu đồng/m², INSTALL_FEE là triệu đồng/bộ sạc. BUDGET = 900.000 triệu đồng = 900 tỷ đồng. Ví dụ 15.110 triệu đồng tương đương 15,11 tỷ đồng, không phải 15.110 đồng.
- Chi phí trạm = diện tích mỗi bộ sạc (evs_parking_area = 15 m²) × tổng số bộ sạc × giá đất + giá thiết bị. Inspector tách mặt bằng, thiết bị và tổng chi phí, kèm quy đổi tỷ đồng. Ngân sách đã dùng = chi phí toàn kế hoạch − chi phí ban đầu + phí di dời; chi phí hạ tầng hiện hữu không tiêu vào ngân sách mở rộng.
- Thời gian chờ: hàng đợi hữu hạn M/M/1/N; ước tính phần chờ sau khi trừ thời gian phục vụ. Cường độ đến dùng demand × 0,08 mỗi nút, là giả định minh họa.
- Bán kính theo công suất hiện tại: r(C) = Rmax / [1 + exp(−C/C₀)], với Rmax = 1 km và C₀ = 1000 × capacity_unit = 1.000 kW. helpers.py lưu capability bằng MW nên cần nhân 1.000 trước khi áp dụng thang kW; demo dùng trực tiếp tổng kW. Ví dụ 60 kW → 515,0 m; 800 kW → 690,0 m; 1.000 kW → 731,1 m. Cả phần tính chỉ số, heuristic lợi ích và vòng phục vụ dùng bán kính tính lại từ cấu hình ở bước đang xem. Đây là hiệu chỉnh đơn vị; có thể làm thay đổi kết quả các lần mô phỏng/huấn luyện mới, không sửa số liệu bài báo đã báo cáo.
- Nhu cầu động dùng các công thức/hệ số trong helpers.py. Khoảng cách sử dụng Haversine, không giải đường đi ngắn nhất trên đồ thị.
- Các cột R, lợi ích, chi phí và công bằng chuẩn hóa theo cùng kế hoạch ban đầu (=100). Score dùng benefit chuẩn hóa, chi phí thành phần chuẩn hóa, fairness gốc và phạt lưới; sau đó chuẩn hóa để hiển thị. Trường hợp score ban đầu âm, phép chuẩn hóa vẫn giữ chiều cao hơn tốt hơn.
- Mọi chuỗi hành động là **kịch bản giả lập xác định**. Tên DQN/GA chỉ xác định kịch bản minh họa riêng; không chạy policy, không tìm kiếm di truyền, không nạp checkpoint. Kết quả so sánh minh họa kế hoạch của demo, không phải bằng chứng xếp hạng các thuật toán.
- Kịch bản DQN/GA bao gồm xây mới, bổ sung và di dời; các Greedy minh họa xây mới theo tiêu chí tương ứng. Khi hành động tiếp theo vượt ngân sách hoặc không còn điều kiện, chuỗi dừng.
- Số liệu gốc trong bài báo và các hình res vẫn được giữ ở trang nghiên cứu, tách khỏi kết quả chạy của app.

## CSV

UTF-8 với BOM, một hàng cho mỗi trạm. Gồm phương pháp, bước, cờ dữ liệu illustrative_no_model, ID trạm/nút, hiện hữu, tọa độ, 12 cột số bộ sạc theo kW, tổng bộ sạc, tổng kW, bán kính km, chờ phút ước tính, bus đấu nối, tải sạc/dư địa MW và quá tải. Tiền tệ ghi rõ cost_unit = million_vnd, station_cost_million_vnd, land_cost_million_vnd, equipment_cost_million_vnd, budget_total_million_vnd, budget_used_million_vnd; diện tích là land_area_m2. Không gộp số liệu của phương án khác hoặc của bước cuối vào CSV khi người dùng đang tua lại.

## Cấu trúc triển khai

- index.html / styles.css / app.js: trang nghiên cứu và điều khiển GIF theo format cũ.
- planner.js: trạng thái demo bổ sung, replay, inspector và tải CSV; khởi tạo khi mở mục demo.
- planner.css: CSS chỉ áp dụng trong .interactive-demo; ID của các control có tiền tố planner- để tránh trùng với bảng và mục phương pháp của bài báo.
- planning-engine.mjs: tính kế hoạch/chỉ số, tạo kịch bản và CSV; không phụ thuộc DOM.
- map-renderer.mjs: SVG ngoại tuyến với lớp nhu cầu, phục vụ, bus và trạm.
- assets/planning-data.json: đầu vào đã xuất sẵn, không chứa mô hình.
- scripts/export_planning_data.py: xuất dữ liệu bằng thư viện Python chuẩn; không import môi trường RL.
- simulation.mjs và scripts/render_animation.py: giữ bộ dựng GIF cho trang nghiên cứu cũ.

Để cập nhật dữ liệu đầu vào:

~~~bash
python demo/scripts/prepare_assets.py
python demo/scripts/export_planning_data.py
~~~

## Kiểm tra

~~~bash
node --test demo/tests/planning-engine.test.mjs demo/tests/simulation.test.mjs
python demo/tests/test_radius_units.py
CHROMEDRIVER=/path/to/chromedriver python demo/tests/browser_smoke.py
~~~

Browser test cần Selenium, Chrome/ChromeDriver và HTTP server cục bộ đang chạy. Kiểm tra 5 trạm đầu vào, bốn phương pháp, các lớp bản đồ, replay/best, chọn bằng bàn phím, inspector 12 mức sạc, hai bản đồ so sánh, CSV tải thật, năm kích thước màn hình, request chỉ tới localhost và lỗi dữ liệu/thử lại. Screenshot nằm trong /tmp.

Dữ liệu đường bộ: © OpenStreetMap contributors. Giữ attribution khi sử dụng lại; các hình/PDF nghiên cứu thuộc nguồn công trình.
