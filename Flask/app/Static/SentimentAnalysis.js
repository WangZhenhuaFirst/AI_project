$(document).ready(function () {
    //要执行的js代码段
    summary()
    echart1()
    table1()
});

function summary() {
    $.ajax({
        url: "/SentimentAnalysis/GetSentimentCount",
        type: "Get",
        data:
        {
        },
        dataType: 'json',
        success: function (data) {
            console.log(data)

            $('#TotalSentiment').html(data.result.TotalSentiment)
            $('#NegativeSentiment').text(data.result.NegativeSentiment)
            $('#PositiveSentiment').text(data.result.PositiveSentiment)

            var myChart = echarts.init(document.getElementById('container2'));
            myChart.setOption({
                title: {
                    text: '舆情分类',
                    x: 'left'
                },
                tooltip: {
                    trigger: 'item',
                    formatter: "{a} <br/>{b} : {c} ({d}%)"
                },
                color: ['#CD5C5C', '#00CED1', '#9ACD32', '#FFC0CB'],
                stillShowZeroSum: false,
                series: [
                    {
                        name: '舆情分类',
                        type: 'pie',
                        radius: '65%',
                        center: ['52%', '50%'],
                        label: {
                            formatter: '{a|{a}}{abg|}\n{hr|}\n  {b|{b}：}{c}  {per|{d}%}  ',
                            backgroundColor: '#eee',
                            borderColor: '#aaa',
                            borderWidth: 1,
                            borderRadius: 4,
                            rich: {
                                a: {
                                    color: '#999',
                                    lineHeight: 22,
                                    align: 'center'
                                },
                                hr: {
                                    borderColor: '#aaa',
                                    width: '100%',
                                    borderWidth: 0.5,
                                    height: 0
                                },
                                b: {
                                    fontSize: 12,
                                    lineHeight: 33
                                },
                                per: {
                                    color: '#eee',
                                    backgroundColor: '#334455',
                                    padding: [2, 4],
                                    borderRadius: 2
                                }
                            }
                        },
                        data: [
                            { value: data.result.NegativeSentiment, name: '负面舆情' },
                            { value: data.result.PositiveSentiment, name: '正面舆情' },
                        ],
                        itemStyle: {
                            emphasis: {
                                shadowBlur: 10,
                                shadowOffsetX: 0,
                                shadowColor: 'rgba(128, 128, 128, 0.5)'
                            }
                        }
                    }
                ]
            })


        },
        error: function (e) {
            $.message({
                message: '处理失败！',
                type: 'error'
            });
        }
    })
}

function echart1() {
    $.ajax({
        url: "/SentimentAnalysis/getSentimentSource",
        type: "Get",
        data:
        {
        },
        dataType: 'json',
        success: function (data) {

            var myChart = echarts.init(document.getElementById('container1'));
            myChart.setOption({
                title: {
                    text: '数据来源',
                    x: 'left'
                },
                tooltip: {
                    trigger: 'item',
                    formatter: "{a} <br/>{b} : {c} ({d}%)"
                },
                color: ['#CD5C5C', '#00CED1', '#9ACD32', '#FFC0CB'],
                stillShowZeroSum: false,
                series: [
                    {
                        name: '数据来源',
                        type: 'pie',
                        radius: '65%',
                        center: ['52%', '50%'],
                        label: {
                            formatter: '{a|{a}}{abg|}\n{hr|}\n  {b|{b}：}{c}  {per|{d}%}  ',
                            backgroundColor: '#eee',
                            borderColor: '#aaa',
                            borderWidth: 1,
                            borderRadius: 4,
                            rich: {
                                a: {
                                    color: '#999',
                                    lineHeight: 22,
                                    align: 'center'
                                },
                                hr: {
                                    borderColor: '#aaa',
                                    width: '100%',
                                    borderWidth: 0.5,
                                    height: 0
                                },
                                b: {
                                    fontSize: 12,
                                    lineHeight: 33
                                },
                                per: {
                                    color: '#eee',
                                    backgroundColor: '#334455',
                                    padding: [2, 4],
                                    borderRadius: 2
                                }
                            }
                        },
                        data: [
                            { value: data.result.MeiTuan, name: '美团' },
                            { value: data.result.DaZhongDianPing, name: '大众点评' },
                        ],
                        itemStyle: {
                            emphasis: {
                                shadowBlur: 10,
                                shadowOffsetX: 0,
                                shadowColor: 'rgba(128, 128, 128, 0.5)'
                            }
                        }
                    }
                ]
            })

        },
        error: function (e) {
            $.message({
                message: '处理失败！',
                type: 'error'
            });
        }
    })
}

function table1() {
    //1.初始化Table
    var oTable = new TableInit();
    oTable.Init();

    //2.初始化Button的点击事件
    //var oButtonInit = new ButtonInit();
    //oButtonInit.Init();
}

var TableInit = function () {
    var oTableInit = new Object();
    //初始化Table
    oTableInit.Init = function () {
        $('#RecentSentiment').bootstrapTable({
            url: '/SentimentAnalysis/getRecentSentiment',         //请求后台的URL（*）
            method: 'get',                      //请求方式（*）
            toolbar: '#toolbar',                //工具按钮用哪个容器
            striped: true,                      //是否显示行间隔色
            cache: false,                       //是否使用缓存，默认为true，所以一般情况下需要设置一下这个属性（*）
            pagination: true,                   //是否显示分页（*）
            sortable: true,                     //是否启用排序
            sortOrder: "asc",                   //排序方式
            queryParams: oTableInit.queryParams,//传递参数（*）
            sidePagination: "server",           //分页方式：client客户端分页，server服务端分页（*）
            pageNumber: 1,                       //初始化加载第一页，默认第一页
            pageSize: 5,                       //每页的记录行数（*）
            pageList: [10, 25, 50, 100],        //可供选择的每页的行数（*）
            search: false,                       //是否显示表格搜索，此搜索是客户端搜索，不会进服务端，所以，个人感觉意义不大
            strictSearch: true,
            showColumns: false,                  //是否显示所有的列
            showRefresh: true,                  //是否显示刷新按钮
            minimumCountColumns: 2,             //最少允许的列数
            clickToSelect: false,                //是否启用点击选中行
            height: 700,                        //行高，如果没有设置height属性，表格自动根据记录条数觉得表格高度
            uniqueId: "reviewId",                     //每一行的唯一标识，一般为主键列
            showToggle: false,                    //是否显示详细视图和列表视图的切换按钮
            cardView: false,                    //是否显示详细视图
            detailView: false,                   //是否显示父子表
            columns: [{
                checkbox: true
            }, {
                field: 'username',
                title: '用户名',
                width: 120,
            }, {
                field: 'resttitle',
                title: '店铺',
                width: 120
            }, {
                field: 'comment',
                title: '评论',
            }, {
                field: 'source',
                title: '来源',
                width: 80,
            }, {
                field: 'sentiment_classification',
                title: '标签',
                width: 80,
                formatter: function (value, row, index) {
                    var a = "";
                    if (value == "正面情感") {
                        var a = '<span style="color:#c12e2a;"><i class="fa fa-times-circle-o" aria-hidden="true"></i>' + value + '</span>';
                    } else if (value == "负面情感") {
                        var a = '<span style="color:#3e8f3e"><i class="fa fa-check-circle-o" aria-hidden="true"></i>' + value + '</span>';
                    }
                    return a;
                },
            },
            {
                field: 'operate',
                title: '操作',
                events: operateEvents,//给按钮注册事件
                formatter: addFunctionAlty,//表格中增加按钮  
                width: 130
            }
            ]
        });
    };

    //得到查询的参数
    oTableInit.queryParams = function (params) {
        var temp = {
        };
        return temp;
    };
    return oTableInit;
};

function timestampToTime(timestamp) {
    var date = new Date(parseInt(timestamp));//时间戳为10位需*1000，时间戳为13位的话不需乘1000
    console.log(date)
    var Y = date.getUTCFullYear() + '-';
    var M = (date.getUTCMonth() + 1 < 10 ? '0' + (date.getUTCMonth() + 1) : date.getUTCMonth() + 1) + '-';
    var D = (date.getUTCDate() < 10 ? '0' + date.getUTCDate() : date.getUTCDate()) + ' ';
    var h = (date.getUTCHours() < 10 ? '0' + date.getUTCHours() : date.getUTCHours()) + ':';
    var m = (date.getUTCMinutes() < 10 ? '0' + date.getUTCMinutes() : date.getUTCMinutes()) + ':';
    var s = (date.getUTCSeconds() < 10 ? '0' + date.getUTCSeconds() : date.getUTCSeconds());
    return Y + M + D + h + m + s;
}

function addFunctionAlty(value, row, index) {
    return [
        '<button id="view" type="button" class="btn btn-default">查看</button>',
        '<button id="analysis" type="button" class="btn btn-default">分析</button>',
    ].join('');
}

window.operateEvents = {
    'click #view': function (e, value, row, index) {
        window.open(row.url)
    }, 'click #analysis': function (e, value, row, index) {
        link = '/SentimentAnalysis/Analysis?reviewId=' + row.reviewId
        window.open(link)
    }
};