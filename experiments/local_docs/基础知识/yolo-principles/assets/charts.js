// YOLO 原理页图表：颜色全部取自 CSS 变量
(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  var grid = { left: 62, right: 28, top: 34, bottom: 46 };
  var axisStyle = {
    axisLine: { lineStyle: { color: rule } },
    axisLabel: { color: muted },
    splitLine: { show: false }
  };
  var yStyle = {
    axisLine: { show: false },
    axisLabel: { color: muted },
    splitLine: { lineStyle: { color: rule } }
  };

  // ---------- 图 1：置信度排序下的 P / R 曲线 ----------
  var elPr = document.getElementById('chart-pr');
  if (elPr && window.echarts) {
    var pr = echarts.init(elPr, null, { renderer: 'svg' });
    // 构造一条合理的 PR 曲线：召回上升，精度衰减
    var recall = [], prec = [], f1 = [];
    for (var i = 0; i <= 40; i += 1) {
      var r = i / 40;
      var p = 0.97 - 0.86 * Math.pow(r, 2.3);
      p = Math.max(p, 0.06);
      recall.push(+r.toFixed(3));
      prec.push(+p.toFixed(3));
      f1.push(+((2 * p * r) / (p + r + 1e-9)).toFixed(3));
    }
    pr.setOption({
      animation: false,
      grid: grid,
      tooltip: { trigger: 'axis', appendToBody: true },
      legend: { data: ['Precision', 'F1'], textStyle: { color: muted }, top: 0, right: 0 },
      xAxis: Object.assign({
        type: 'category', data: recall,
        name: 'Recall', nameLocation: 'middle', nameGap: 28,
        axisLabel: { color: muted, interval: 7 }
      }, { axisLine: axisStyle.axisLine, splitLine: axisStyle.splitLine }),
      yAxis: Object.assign({ type: 'value', name: 'Precision', min: 0, max: 1 }, yStyle),
      series: [
        {
          name: 'Precision', type: 'line', data: prec, showSymbol: false, smooth: true,
          lineStyle: { color: accent, width: 2.4 },
          areaStyle: { color: accent + '22' }
        },
        {
          name: 'F1', type: 'line', data: f1, showSymbol: false, smooth: true,
          lineStyle: { color: accent2, width: 1.8, type: 'dashed' }
        }
      ]
    });
    window.addEventListener('resize', function () { pr.resize(); });
  }

  // ---------- 图 2：IoU 阈值升高时，小目标可匹配率骤降 ----------
  var elIou = document.getElementById('chart-iou-size');
  if (elIou && window.echarts) {
    var ci = echarts.init(elIou, null, { renderer: 'svg' });
    var ths = [], big = [], mid = [], small = [];
    // 模型偏差固定为 2px。框边长 s 时，平移 2px 的一维 IoU 近似 (s-2)/(s+2)
    [64, 32, 16, 8].forEach(function (s) {
      var oneD = (s - 2) / (s + 2);
      var iou2d = Math.max(0, Math.pow(oneD, 2));
      ths.push(s);
      if (s === 64) big.push(+(iou2d * 100).toFixed(1));
      if (s === 32) mid.push(+(iou2d * 100).toFixed(1));
      if (s === 16) small.push(+(iou2d * 100).toFixed(1));
    });
    var dataByEdge = [64, 32, 16, 8].map(function (s) {
      var oneD = (s - 2) / (s + 2);
      return [s, +(Math.max(0, Math.pow(oneD, 2)) * 100).toFixed(1)];
    });
    ci.setOption({
      animation: false,
      grid: grid,
      tooltip: { trigger: 'axis', appendToBody: true, valueFormatter: function (v) { return v + '%'; } },
      xAxis: Object.assign({
        type: 'category', data: ['8 px', '16 px', '32 px', '64 px'],
        name: '目标边长', nameLocation: 'middle', nameGap: 28
      }, { axisLine: axisStyle.axisLine, splitLine: axisStyle.splitLine }),
      yAxis: Object.assign({ type: 'value', name: 'IoU %', max: 100 }, yStyle),
      series: [{
        type: 'bar', barWidth: '48%',
        data: [dataByEdge[3][1], dataByEdge[2][1], dataByEdge[1][1], dataByEdge[0][1]],
        itemStyle: {
          color: function (p) { return p.dataIndex < 2 ? accent2 : accent; },
          borderRadius: [3, 3, 0, 0]
        },
        label: { show: true, position: 'top', color: ink, formatter: '{c}%' }
      }]
    });
    window.addEventListener('resize', function () { ci.resize(); });
  }

  // ---------- 图 3：mAP50 与 mAP50-95 的关系（10 个阈值上的 AP） ----------
  var elMap = document.getElementById('chart-map-thresholds');
  if (elMap && window.echarts) {
    var mt = echarts.init(elMap, null, { renderer: 'svg' });
    var thsArr = [], apsArr = [];
    for (var t = 0; t <= 9; t += 1) {
      var th = 0.5 + t * 0.05;
      // 经验衰减：IoU 要求越高 AP 越低，后期加速下降
      var ap = 0.72 - 0.55 * Math.pow((th - 0.5) / 0.45, 1.7);
      thsArr.push(th.toFixed(2));
      apsArr.push(+Math.max(ap, 0.03).toFixed(3));
    }
    mt.setOption({
      animation: false,
      grid: grid,
      tooltip: { trigger: 'axis', appendToBody: true, valueFormatter: function (v) { return v; } },
      xAxis: Object.assign({
        type: 'category', data: thsArr,
        name: 'IoU 阈值', nameLocation: 'middle', nameGap: 28,
        axisLabel: { color: muted, interval: 1 }
      }, { axisLine: axisStyle.axisLine, splitLine: axisStyle.splitLine }),
      yAxis: Object.assign({ type: 'value', name: 'AP', min: 0, max: 0.8 }, yStyle),
      series: [{
        type: 'bar', barWidth: '56%', data: apsArr,
        itemStyle: { color: accent, borderRadius: [3, 3, 0, 0] },
        markLine: {
          symbol: 'none', silent: true,
          lineStyle: { color: accent2, type: 'dashed', width: 1.3 },
          label: { color: accent2, formatter: 'mAP50 = 该柱', position: 'insideEndTop' },
          data: [{ type: 'average', name: '均值' }]
        }
      }]
    });
    window.addEventListener('resize', function () { mt.resize(); });
  }

  // ---------- 图 4：置信度阈值对 P / R 的权衡 ----------
  var elConf = document.getElementById('chart-conf-tradeoff');
  if (elConf && window.echarts) {
    var ct = echarts.init(elConf, null, { renderer: 'svg' });
    var xs = [], ps = [], rs = [];
    for (var k = 1; k <= 20; k += 1) {
      var c = k / 20;
      xs.push(c.toFixed(2));
      // 阈值提高：精度上升、召回下降
      ps.push(+(0.35 + 0.62 * Math.pow(c, 0.55)).toFixed(3));
      rs.push(+(0.95 - 0.88 * Math.pow(c, 1.9)).toFixed(3));
    }
    ct.setOption({
      animation: false,
      grid: grid,
      tooltip: { trigger: 'axis', appendToBody: true },
      legend: { data: ['Precision', 'Recall'], textStyle: { color: muted }, top: 0, right: 0 },
      xAxis: Object.assign({
        type: 'category', data: xs,
        name: 'conf 阈值', nameLocation: 'middle', nameGap: 28,
        axisLabel: { color: muted, interval: 3 }
      }, { axisLine: axisStyle.axisLine, splitLine: axisStyle.splitLine }),
      yAxis: Object.assign({ type: 'value', min: 0, max: 1 }, yStyle),
      series: [
        { name: 'Precision', type: 'line', data: ps, showSymbol: false, smooth: true, lineStyle: { color: accent, width: 2.3 } },
        { name: 'Recall', type: 'line', data: rs, showSymbol: false, smooth: true, lineStyle: { color: accent2, width: 2.3 } }
      ]
    });
    window.addEventListener('resize', function () { ct.resize(); });
  }
})();
