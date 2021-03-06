## Flask文件配置
- 复制新闻文件和model文件到Flask\app\Static文件夹
- 配置Flask\app\models\Config.py中的新闻和model名字

## Flask 调试：
- 终端打开Flask文件夹，执行python -m flask run

## Flask部分的文件结构
Flask/app/templates/Index.html 是主要的页面；
点击其中的按钮，事件会触发Flask/app/Static/Index.js中的函数；
这些函数，会再触发flask后端，即Flask/app/main/routes.py中的函数；
这些函数，再触发各个类中的方法。

## Flask 运行：
- 只需要将app文件夹上传到web服务器 Ubuntu目录
- command to run：
    sudo systemctl start flaskapp
 to check running status:
    sudo systemctl status flaskapp

the service contents for flaskapp
    /etc/systemd/system/flaskapp.service
===============
[Unit]
Description=gunicorn daemon
After=network.target

[Service]
PIDFile=/run/gunicorn/pid
User=ubuntu
Group=www-data
RuntimeDirectory=gunicorn
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/.local/bin/gunicorn --pid /run/gunicorn/pid   --workers 3 \
           --bind 0.0.0.0:5000 --workers 3 "app:create_app('')"
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s TERM $MAINPID
PrivateTmp=true

[Install]
WantedBy=multi-user.target
============





