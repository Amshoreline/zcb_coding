# Docker基本使用

- docker_hub: https://hub.docker.com/r/pytorch/pytorch/tags?page=1&ordering=last_updated

- run

    ```python
    docker run docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
    Options:
        -d: 后台
        -i: 交互（和-t一起用）
        -t: 输入终端（和-i一起用）
        -p: 主机(宿主)端口:容器端口
        --name: 名称
        --runtime: --runtime nvidia
        --privileged=true: 给docker root权限
    
    docker run -dit --name zcb_first --runtime nvidia --privileged=true -p 18822:22 -p 18880:8880 -p 18881:8881 -p 18882:8882 -p 18883:8883 -p 18884:8884 -p 18885:8885 -p 18886:8886 -p 18887:8887 -p 18888:8888 -p 18889:8889 -p 18890:8890 -p 18891:8891 -p 18892:8892 -p 18893:8893 -p 18894:8894 -p 18895:8895 -p 18896:8896 -p 18897:8897 -p 18898:8898 -p 18899:8899 zcb_image:1.1
    ```

- start/stop

    ```python
    docker stop CONTAINER
    dokcer start CONTAINRE
    ```

- attach

    ```python
    进入：docker attach CONTAINER
    退出：ctrl + p, ctrl + q
    ```

- commit&save&

    ```python
    docker commit [OPTIONS] CONTAINER [REPOSITORY[:TAG]]
    
    docker commit zcb_first zcb_image:1.1

    docker save zcb_image:1.1 > zcb_image_1.1.tar

    docker load --input zcb_image_1.1.tar
    ```

- ssh
    开启ssh服务

    ```python
    vim /etc/ssh/sshd_config 
    set: PermitRootLogin yes
    ```

- neo4j

    ```python
    修改\etc\neo4j\neo4j.conf
    dbms.default_database=neo4j 可以指定不同的数据库
    dbms.default_listen_address=0.0.0.0 用于开放外部访问
    dbms.connector.http.listen_address=:8888 用于开放网页访问端口
    dbms.connector.bolt.listen_address=:8889 在网页访问后开放数据库控制端口
    ```

    镜像用法

    ```python
    docker pull neo4j:4.2.3
    docker run -p 7474:7474 -p 7687:7687 -v $HOME/neo4j/data:/data neo4j:4.2.3
    这里的 7474 是HTTP端口，7687 是Bolt端口。v这里是把container的存储位置放在外边。
    可以再加上 --env=NEO4J_AUTH=none 关掉认证功能。
    ```