$(document).ready(function () {
    //要执行的js代码段
    Analysis()
});

function textchanged() {
    Analysis()
}

function Analysis() {
    $.message({
        message: '处理中',
        type: 'success'
    });
    $.ajax({
        url: "/SentimentAnalysis/GetSentimentAnalysis",
        type: "Get",
        data:
        {
            Comment: $("#Comment").val(),
        },
        dataType: 'json',
        success: function (data) {
            console.log(data)
            var record = Array()
            var name = Array()
            var value = Array()
            data.result.data.forEach(key => {
                if (key[1] > 0) {
                    record.push({
                        name: key[0],
                        value: key[1]
                    });
                }
                name.push(key[0])
                value.push(key[1])
            })

            var myChart1 = echarts.init(document.getElementById('container1'));
            myChart1.setOption(
                {
                    tooltip: {},
                    series: [{
                        type: 'wordCloud',
                        gridSize: 2,
                        sizeRange: [12, 50],
                        rotationRange: [-90, 90],
                        shape: 'pentagon',
                        width: 600,
                        height: 400,
                        drawOutOfBound: true,
                        textStyle: {
                            normal: {
                                color: function () {
                                    return 'rgb(' + [
                                        Math.round(Math.random() * 160),
                                        Math.round(Math.random() * 160),
                                        Math.round(Math.random() * 160)
                                    ].join(',') + ')';
                                }
                            },
                            emphasis: {
                                shadowBlur: 10,
                                shadowColor: '#333'
                            }
                        },
                        data: record
                    }]
                }
            )

            var myChart2 = echarts.init(document.getElementById('container2'));
            myChart2.setOption(
                {
                    tooltip: {
                    },
                    angleAxis: {
                        type: 'category',
                        data: name,
                        axisLabel: {
                            textStyle: {
                                color: '#666',//坐标值得具体的颜色
                                fontSize: 10,
                            }
                        }
                    },
                    radiusAxis: {
                    },
                    polar: {
                    },
                    series: [{
                        type: 'bar',
                        data: value,
                        coordinateSystem: 'polar',
                        name: '文本细粒度情感分析',
                        stack: 'a'
                    },
                    ],
                    legend: {
                        show: true,
                        data: ['文本细粒度情感分析']
                    }
                }
            )
            $.message({
                message: '处理成功',
                type: 'success'
            });
        },
        error: function (e) {
            $.message({
                message: '处理失败！',
                type: 'error'
            });
        }
    })
}