widgets/qparameterwidget_ui.py: ui/qparameterwidget.ui
	pyuic6 -o $@ $<

main_ui.py: ui/main.ui
	pyuic6 -o $@ $<

all: \
	widgets/qparameterwidget_ui.py \
	main_ui.py ;
