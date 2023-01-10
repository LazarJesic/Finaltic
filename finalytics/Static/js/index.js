import myJson from './stocks.json' assert {type: 'json'};

let searchable = myJson.stocks
//DOM reference

const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.content');



tabs.forEach((tab, tabIndex) => {
    tab.addEventListener('click', handleTab(tabContents[tabIndex]));
});

function handleTab(tabContent) {
    let show = true;
    return () => {
        tabContent.style.display = show ? 'block' : 'none';
        show = !show;
    }
}

