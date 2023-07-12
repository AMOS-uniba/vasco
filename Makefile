all: \
	widgets/qparameterwidget_ui.py \
	main_ui.py ;

widgets/qparameterwidget_ui.py: ui/wx_parameter.ui
	pyuic6 -o $@ $<

main_ui.py: ui/main.ui
	pyuic6 -o $@ $<
