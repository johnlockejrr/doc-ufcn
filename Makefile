.PHONY: release

release:
	$(eval version:=$(shell cat VERSION))
	git commit VERSION -m "Version $(version)"
	git tag $(version)
	git push origin main $(version)
