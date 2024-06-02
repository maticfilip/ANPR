import pkg_resources

vrzija=pkg_resources.working_set.by_key['pytesseract'].version

print(vrzija)