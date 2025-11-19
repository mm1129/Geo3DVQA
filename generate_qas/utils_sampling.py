import numpy as np

def filter_by_distance(points, min_dist=30):
    """
    位置が近すぎる選択肢を排除するフィルタ関数
    
    Args:
        points (list): 各候補の画素座標（中心点）のリスト [(y, x) | (y,x,h,w)]
        min_dist (float): 2点間のユークリッド距離がこの値未満なら同一クラスタとみなして間引く
    
    Returns:
        list: 距離条件を満たすよう間引いたpoints
    """
    kept = []
    for p in points:
        cy, cx = p[:2]
        if all(np.hypot(cy - ky, cx - kx) >= min_dist for ky, kx, *_ in kept):
            kept.append(p)
    return kept

def filter_by_metric_gap(points, metrics, min_gap=0.08, target_count=4):
    """
    指標値の差が十分にある選択肢を選ぶフィルタ関数
    
    Args:
        points (list): 候補点のリスト
        metrics (np.ndarray): 各候補点の指標値
        min_gap (float): 指標値の最小差
        target_count (int): 目標とする選択肢の数
    
    Returns:
        tuple: (フィルタリングされたpoints, フィルタリングされたmetrics)
    """
    if len(points) == 0:
        return [], np.array([])
        
    # 値の大きい順に並べ替え
    order = metrics.argsort()[::-1]
    points = np.array(points)[order]
    metrics = metrics[order]
    
    # 値が十分離れているかチェック
    keep = [points[0]]
    keep_metrics = [metrics[0]]
    
    for p, m in zip(points[1:], metrics[1:]):
        if all(abs(m - k) >= min_gap for k in keep_metrics):
            keep.append(p)
            keep_metrics.append(m)
        if len(keep) == target_count:
            break
    
    # もし足りなければ追加（配列境界チェック付き）
    while len(keep) < target_count and len(keep) < len(points):
        next_index = len(keep)
        if next_index < len(points) and next_index < len(metrics):
            keep.append(points[next_index])
            keep_metrics.append(metrics[next_index])
        else:
            break  # 配列境界を超える場合は安全に終了
    
    return keep, np.array(keep_metrics)

def select_by_quartiles(points, metrics, target_count=4, min_points_per_quartile=1):
    """
    四分位数に基づいて選択肢を選ぶ関数
    
    Args:
        points (list): 候補点のリスト
        metrics (np.ndarray): 各候補点の指標値
        target_count (int): 目標とする選択肢の数
        min_points_per_quartile (int): 各四分位数から最低限選ぶ点数
    
    Returns:
        tuple: (選択されたpoints, 選択されたmetrics)
    """
    if len(points) == 0:
        return [], np.array([])
    
    # 指標値の四分位数を計算
    quartiles = np.percentile(metrics, [0, 25, 50, 75, 100])
    
    # 各四分位数の範囲に属する点を分類
    quartile_points = [[] for _ in range(4)]
    quartile_metrics = [[] for _ in range(4)]
    
    for i, (p, m) in enumerate(zip(points, metrics)):
        for q in range(4):
            if quartiles[q] <= m <= quartiles[q+1]:
                quartile_points[q].append(p)
                quartile_metrics[q].append(m)
                break
    
    # 各四分位数から最低限の点数を選ぶ
    selected_points = []
    selected_metrics = []
    
    for q in range(4):
        if len(quartile_points[q]) >= min_points_per_quartile:
            # 四分位数内でランダムに選択
            indices = np.random.choice(len(quartile_points[q]), 
                                     min(min_points_per_quartile, len(quartile_points[q])), 
                                     replace=False)
            selected_points.extend([quartile_points[q][i] for i in indices])
            selected_metrics.extend([quartile_metrics[q][i] for i in indices])
    
    # 選択された点が少ない場合は、残りの点からランダムに追加
    if len(selected_points) < target_count:
        remaining_points = []
        remaining_metrics = []
        
        for q in range(4):
            for i, p in enumerate(quartile_points[q]):
                if p not in selected_points:
                    remaining_points.append(p)
                    remaining_metrics.append(quartile_metrics[q][i])
        
        if remaining_points:
            indices = np.random.choice(len(remaining_points), 
                                     min(target_count - len(selected_points), len(remaining_points)), 
                                     replace=False)
            selected_points.extend([remaining_points[i] for i in indices])
            selected_metrics.extend([remaining_metrics[i] for i in indices])
    
    # 距離フィルタを適用
    selected_points = filter_by_distance(selected_points, min_dist=30)
    selected_metrics = np.array(selected_metrics)[:len(selected_points)]
    
    return selected_points, selected_metrics 

def select_by_quartiles_with_top(points, metrics, target_count=4, min_points_per_quartile=1):
    """
    四分位数に基づいて選択肢を選ぶ関数（最高値を必ず含む）
    
    Args:
        points (list): 候補点のリスト
        metrics (np.ndarray): 各候補点の指標値
        target_count (int): 目標とする選択肢の数
        min_points_per_quartile (int): 各四分位数から最低限選ぶ点数
    
    Returns:
        tuple: (選択されたpoints, 選択されたmetrics)
    """
    if len(points) == 0:
        return [], np.array([])
    
    # 最高値のインデックスを最初に確保
    top_idx = np.argmax(metrics)
    top_point = points[top_idx]
    top_metric = metrics[top_idx]
    
    # 最高値を除外した残りのデータ
    remaining_points = points.copy()
    remaining_metrics = metrics.copy()
    remaining_points.pop(top_idx)
    remaining_metrics = np.delete(remaining_metrics, top_idx)
    
    # 残りのtarget_count-1点を四分位で選択
    if len(remaining_points) == 0:
        return [top_point], np.array([top_metric])
    
    # 指標値の四分位数を計算
    quartiles = np.percentile(remaining_metrics, [0, 25, 50, 75, 100])
    
    # 各四分位数の範囲に属する点を分類
    quartile_points = [[] for _ in range(4)]
    quartile_metrics = [[] for _ in range(4)]
    
    for i, (p, m) in enumerate(zip(remaining_points, remaining_metrics)):
        for q in range(4):
            if quartiles[q] <= m <= quartiles[q+1]:
                quartile_points[q].append(p)
                quartile_metrics[q].append(m)
                break
    
    # 各四分位数から選ぶ点数を計算
    remaining_count = min(target_count - 1, len(remaining_points))
    points_per_quartile = max(min_points_per_quartile, remaining_count // 4)
    
    # 各四分位数から点を選ぶ
    selected_points = []
    selected_metrics = []
    
    for q in range(4):
        if len(quartile_points[q]) >= min_points_per_quartile:
            # 四分位数内でランダムに選択
            count = min(points_per_quartile, len(quartile_points[q]))
            indices = np.random.choice(len(quartile_points[q]), count, replace=False)
            selected_points.extend([quartile_points[q][i] for i in indices])
            selected_metrics.extend([quartile_metrics[q][i] for i in indices])
    
    # 選択された点が足りない場合は、残りの点からランダムに追加
    if len(selected_points) < remaining_count:
        # まだ選ばれていない点を特定
        all_selected = set(selected_points)
        unselected_points = []
        unselected_metrics = []
        
        for q in range(4):
            for i, p in enumerate(quartile_points[q]):
                if p not in all_selected:
                    unselected_points.append(p)
                    unselected_metrics.append(quartile_metrics[q][i])
        
        if unselected_points:
            needed = remaining_count - len(selected_points)
            count = min(needed, len(unselected_points))
            indices = np.random.choice(len(unselected_points), count, replace=False)
            selected_points.extend([unselected_points[i] for i in indices])
            selected_metrics.extend([unselected_metrics[i] for i in indices])
    
    # 最高値を追加
    selected_points.append(top_point)
    selected_metrics.append(top_metric)
    
    # 距離フィルタを適用（最高値を優先して保持）
    # まず最高値をリストの先頭に移動
    sel_points = [top_point] + [p for p in selected_points if p != top_point]
    sel_metrics = [top_metric] + [m for m in selected_metrics if m != top_metric]
    
    # 距離フィルタを適用
    filtered_points = filter_by_distance(sel_points, min_dist=30)
    filtered_metrics = []
    
    # フィルタ後の点に対応するメトリクスを収集
    for p in filtered_points:
        idx = sel_points.index(p)
        filtered_metrics.append(sel_metrics[idx])
    
    return filtered_points, np.array(filtered_metrics) 