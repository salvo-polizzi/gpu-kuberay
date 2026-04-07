{{- define "vscode-rayjob.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "vscode-rayjob.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "vscode-rayjob.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "vscode-rayjob.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "vscode-rayjob.labels" -}}
helm.sh/chart: {{ include "vscode-rayjob.chart" . }}
app.kubernetes.io/name: {{ include "vscode-rayjob.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "vscode-rayjob.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vscode-rayjob.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "vscode-rayjob.pvcName" -}}
{{- printf "%s-shared" (include "vscode-rayjob.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "vscode-rayjob.serviceAccountName" -}}
{{- printf "%s-rayjob-submit" (include "vscode-rayjob.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "vscode-rayjob.rayJobTemplateName" -}}
{{- printf "%s-rayjob-template" (include "vscode-rayjob.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
