$(document).ready(function () {
    //要执行的js代码段
    table1()
});

function table1() {
    //1.初始化Table
    var oTable = new TableInit();
    oTable.Init();
}

var TableInit = function () {
    var oTableInit = new Object();
    //初始化Table
    oTableInit.Init = function () {
        $('#SentimentSearch').bootstrapTable({
            url: '/SentimentAnalysis/GetSentimentAnalysisSearchResult',         //请求后台的URL（*）
            method: 'get',                      //请求方式（*）
            toolbar: '#toolbar',                //工具按钮用哪个容器
            striped: true,                      //是否显示行间隔色
            cache: false,                       //是否使用缓存，默认为true，所以一般情况下需要设置一下这个属性（*）
            pagination: true,                   //是否显示分页（*）
            sortable: false,                     //是否启用排序
            sortOrder: "asc",                   //排序方式
            queryParams: oTableInit.queryParams,//传递参数（*）
            sidePagination: "server",           //分页方式：client客户端分页，server服务端分页（*）
            pageNumber: 1,                       //初始化加载第一页，默认第一页
            pageSize: 10,                       //每页的记录行数（*）
            pageList: [10, 25, 50, 100],        //可供选择的每页的行数（*）
            search: false,                       //是否显示表格搜索，此搜索是客户端搜索，不会进服务端，所以，个人感觉意义不大
            strictSearch: true,
            showColumns: true,                  //是否显示所有的列
            showRefresh: true,                  //是否显示刷新按钮
            minimumCountColumns: 2,             //最少允许的列数
            clickToSelect: true,             //是否启用点击选中行
            height: 800,                        //行高，如果没有设置height属性，表格自动根据记录条数觉得表格高度
            uniqueId: "reviewId",                     //每一行的唯一标识，一般为主键列
            showToggle: true,                    //是否显示详细视图和列表视图的切换按钮
            cardView: false,                    //是否显示详细视图
            detailView: false,                   //是否显示父子表
            columns: [{
                checkbox: true
            }, {
                field: 'username',
                title: '用户名',
                width: 120
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
                width: 80
            }, 
            //{
                //field: 'url',
                //title: 'Url',
                //width: 10
            //},
            {
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
            ],
            fixedColumns: true,
            fixedNumber: 5
        });
    };

    //得到查询的参数
    oTableInit.queryParams = function (params) {
        var temp = {
            limit: params.limit,   //页面大小
            offset: params.offset,  //页码
            KeyWord: $("#KeyWord").val(),
        };
        return temp;
    };
    return oTableInit;
};

function LoadData() {
    $("#SentimentSearch").bootstrapTable('refresh');
};

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