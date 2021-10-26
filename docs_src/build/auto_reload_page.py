#!/bin/python

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from os.path import abspath
from watchFiles import watchFiles

def openWebsite(path):
    chromedriver = '/usr/bin/chromedriver'
    options = webdriver.ChromeOptions()
 #   options.add_argument('window-size=1200x600') # optional
    options.add_argument("--disable-infobars") # removes a warning
    #options.add_argument("--sandbox") # removes a warning
    #options.add_argument("--kiosk")#open in full screen
    

    desired_capabilities = DesiredCapabilities.CHROME.copy()
    

    browser = webdriver.Chrome(executable_path=chromedriver, 
                                options=options, 
                                desired_capabilities=desired_capabilities)
    browser.get(path)

    return browser


path = abspath('../../docs/index.html')
browser = openWebsite("file://" + path)


def callback():
    print("Refresh page")
    browser.refresh()

watchFiles("../../docs/",callback)


browser.quit()
