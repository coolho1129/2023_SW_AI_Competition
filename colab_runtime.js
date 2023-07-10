function ClickConnect(){
    console.log("1분마다 코랩 연결 끊김 방지"); 
    document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(ClickConnect, 1000 * 60);
