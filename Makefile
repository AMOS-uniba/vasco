MAKEFLAGS += --no-builtin-rules --no-builtin-variables

all: \
	vasco/widgets/qparameterwidget_ui.py \
	vasco/main_ui.py ;

vasco/widgets/qparameterwidget_ui.py: vasco/ui/qparameterwidget.ui
	pyuic6 -o $@ $<

vasco/main_ui.py: vasco/ui/main.ui
	pyuic6 -o $@ $<
